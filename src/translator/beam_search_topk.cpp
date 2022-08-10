/* All or part of this file was contributed by NVIDIA under license:
 *   Copyright (C) 2020 NVIDIA Corporation
 *   SPDX-License-Identifier: MIT
 */

#include "translator/beam_search.h"
#include "tensors/tensor_allocator.h"

#include "data/factored_vocab.h"
#include "translator/helpers.h"
#include "translator/nth_element.h"
#include "data/shortlist.h"

namespace marian {

// combine new expandedPathScores and previous beams into new set of beams
Beams BeamSearch::toHyps(const std::vector<uint64_t>& nBestKeys, // [currentDimBatch, beamSize] flattened -> ((batchIdx, beamHypIdx) flattened, word idx) flattened
                         const std::vector<float>& nBestPathScores,  // [currentDimBatch, beamSize] flattened
                         const size_t nBestBeamSize, // for interpretation of nBestKeys
                         const size_t vocabSize,     // ditto.
                         const Beams& beams,
                         const std::vector<Ptr<ScorerState /*const*/>>& states,
                         Ptr<data::CorpusBatch /*const*/> batch, // for alignments only
                         Ptr<FactoredVocab/*const*/> factoredVocab, size_t factorGroup,
                         const std::vector<bool>& dropBatchEntries, // [origDimBatch] - empty source batch entries are marked with true, should be cleared after first use.
                         const std::vector<IndexType>& batchIdxMap) const { // [origBatchIdx -> currentBatchIdx]
  std::vector<float> align; // collects alignment information from the last executed time step
  if(options_->hasAndNotEmpty("alignment") && factorGroup == 0)
    align = scorers_[0]->getAlignment(); // [beam depth * max src length * current batch size] -> P(s|t); use alignments from the first scorer, even if ensemble,

  const auto origDimBatch = beams.size(); // see function search for definition of origDimBatch and currentDimBatch etc.
  Beams newBeams(origDimBatch);           // return value of this function goes here. There are always origDimBatch beams.

  // create a reverse batchMap to obtain original batchIdx in the starting batch size
  // and calculate the current batch size based on non-empty beams
  std::vector<IndexType> reverseBatchIdxMap; // empty if not purging batch entries
  size_t currentDimBatch = beams.size();
  if(PURGE_BATCH) {
    reverseBatchIdxMap.resize(batchIdxMap.size()); // adjust size if doing batch purging.
    currentDimBatch = 0;
    for(int i = 0; i < batchIdxMap.size(); ++i) {
      reverseBatchIdxMap[batchIdxMap[i]] = i; // reverse batch index mapping, multiple occurences get overwritten with the last one,
                                              // which is expected due to down-shifting
      if(!beams[i].empty())
        currentDimBatch++;
    }
  }

  // Hold the flattened logit indices for each state so we can batch retrieval later. Additionally, store the original batch index to we can update the hypothesis in new beams
  std::vector<size_t> origBatchIndices;
  std::vector<size_t> oldBeamHypIndices;
  std::vector<size_t> newBeamHypIndices;
  std::vector<std::vector<uint64_t>> flattenedLogitIndices(states.size()*2);
  
  for(size_t i = 0; i < nBestKeys.size(); ++i) { // [currentDimBatch, beamSize] flattened
    // Keys encode batchIdx, beamHypIdx, and word index in the entire beam.
    // They can be between 0 and (vocabSize * nBestBeamSize * batchSize)-1.
    // (beamHypIdx refers to the GPU tensors, *not* the beams[] array; they are not the same in case of purging)
    const auto  key = nBestKeys[i];
    
    // decompose key into individual indices (batchIdx, beamHypIdx, wordIdx)
    const auto beamHypIdx      = ((key / vocabSize) / vocabSize) % nBestBeamSize;
    const auto currentBatchIdx = ((key / vocabSize) / vocabSize) / nBestBeamSize;
    const auto origBatchIdx    = reverseBatchIdxMap.empty() ? currentBatchIdx : reverseBatchIdxMap[currentBatchIdx]; // map currentBatchIdx back into original position within starting maximal batch size, required to find correct beam

    bool dropHyp = !dropBatchEntries.empty() && dropBatchEntries[origBatchIdx] && factorGroup == 0;
    
    WordIndex firstWordIdx, secondWordIdx;
    if(dropHyp) { // if we force=drop the hypothesis, assign EOS, otherwise the expected word id.
      if(factoredVocab) { // when using factoredVocab, extract the EOS lemma index from the word id, we predicting factors one by one here, hence lemma only
        std::vector<size_t> eosFactors;
        factoredVocab->word2factors(factoredVocab->getEosId(), eosFactors);
        firstWordIdx = (WordIndex)eosFactors[0];
        secondWordIdx = firstWordIdx;
      } else { // without factoredVocab lemma index and word index are the same. Safe cruising. 
        firstWordIdx = trgVocab_->getEosId().toWordIndex();
        secondWordIdx = firstWordIdx;
      }
    } else { // we are not dropping anything, just assign the normal index
      firstWordIdx  = (WordIndex)((key / vocabSize) % vocabSize);
      secondWordIdx = (WordIndex)(key % vocabSize);
    }

    // @TODO: We currently assign a log probability of 0 to all beam entries of the dropped batch entry, instead it might be a good idea to use
    // the per Hyp pathScore without the current expansion (a bit hard to obtain). 
    // For the case where we drop empty inputs, 0 is fine. For other use cases like a forced stop, the penultimate pathScore might be better. 
    // For the empty hyp this would naturally result in 0, too. 
    const float pathScore = dropHyp ? 0.f : nBestPathScores[i]; // 0 (Prob = 1, maximum score) if dropped or expanded path score for (batchIdx, beamHypIdx, word)

    const auto& beam = beams[origBatchIdx];
    auto& newBeam = newBeams[origBatchIdx]; // extended hypotheses are going to be placed in this new beam

    if(newBeam.size() >= beam.size()) // getNBestList() generates N for all batch entries incl. those that already have a narrower beam
      continue;
    if(pathScore == INVALID_PATH_SCORE) // (dummy slot or word that cannot be expanded by current factor)
      continue;
    
    ABORT_IF(pathScore < INVALID_PATH_SCORE, "Actual pathScore ({}) is lower than INVALID_PATH_SCORE ({})??", pathScore, INVALID_PATH_SCORE); // This should not happen in valid situations. Currently the only smaller value would be -inf (effect of overflow in summation?)
    ABORT_IF(beamHypIdx >= beam.size(), "Out of bounds beamHypIdx??"); // effectively this is equivalent to ABORT_IF(beams[origBatchIdx].empty(), ...)

    // map wordIdx to word
    auto prevBeamHypIdx = beamHypIdx; // back pointer
    auto prevHyp = beam[prevBeamHypIdx];
    Word firstWord, secondWord;
    // If short list has been set, then wordIdx is an index into the short-listed word set,
    // rather than the true word index.
    auto shortlist = scorers_[0]->getShortlist();
    if (factoredVocab) {
      // For factored decoding, the word is built over multiple decoding steps,
      // starting with the lemma, then adding factors one by one.
      if (factorGroup == 0) {
        firstWord = factoredVocab->lemma2Word(shortlist ? shortlist->reverseMap(firstWordIdx) : firstWordIdx); // @BUGBUG: reverseMap is only correct if factoredVocab_->getGroupRange(0).first == 0
        secondWord = factoredVocab->lemma2Word(shortlist ? shortlist->reverseMap(secondWordIdx) : secondWordIdx); // @BUGBUG: reverseMap is only correct if factoredVocab_->getGroupRange(0).first == 0
        std::vector<size_t> firstFactorIndices; factoredVocab->word2factors(firstWord, firstFactorIndices);
        std::vector<size_t> secondFactorIndices; factoredVocab->word2factors(secondWord, secondFactorIndices);

        //LOG(info, "{} + {} ({}) -> {} -> {}",
        //    factoredVocab->decode(prevHyp->tracebackWords()),
        //    factoredVocab->word2string(word), factorIndices[0], prevHyp->getPathScore(), pathScore);
      }
      else {
        //LOG(info, "{} |{} ({}) = {} ({}) -> {} -> {}",
        //    factoredVocab->decodeForDiagnostics(beam[beamHypIdx]->tracebackWords()),
        //    factoredVocab->getFactorGroupPrefix(factorGroup), factorGroup,
        //    factoredVocab->getFactorName(factorGroup, wordIdx), wordIdx,
        //    prevHyp->getPathScore(), pathScore);
        firstWord = beam[beamHypIdx]->getWord();
        secondWord = beam[beamHypIdx]->getSecondWord();
        ABORT_IF(!factoredVocab->canExpandFactoredWord(secondWord, factorGroup),
                  "A word without this factor snuck through to here??");
        firstWord = factoredVocab->expandFactoredWord(firstWord, factorGroup, firstWordIdx);
        secondWord = factoredVocab->expandFactoredWord(secondWord, factorGroup, secondWordIdx);
        prevBeamHypIdx = prevHyp->getPrevStateIndex();
        prevHyp = prevHyp->getPrevHyp(); // short-circuit the backpointer, so that the traceback does not contain partially factored words
      }
    }
    else if (shortlist) {
      firstWord = Word::fromWordIndex(shortlist->reverseMap(firstWordIdx));
      secondWord = Word::fromWordIndex(shortlist->reverseMap(secondWordIdx));
    }
    else {
      firstWord = Word::fromWordIndex(firstWordIdx);
      secondWord = Word::fromWordIndex(secondWordIdx);
    }

    auto hyp = Hypothesis::New(prevHyp, firstWord , secondWord, prevBeamHypIdx, pathScore);

    // Set score breakdown for n-best lists
    if(options_->get<bool>("n-best")) {
      ABORT_IF(factoredVocab && factorGroup > 0 && !factoredVocab->canExpandFactoredWord(secondWord, factorGroup),
               "A word without this factor snuck through to here??");
      for(uint64_t j = 0; j < states.size(); ++j) {
        auto lval = states[j]->getLogProbs().getFactoredLogitsTensor(factorGroup); // [maxBeamSize, 2, currentDimBatch, dimFactorVocab]
        // The flatting happens based on actual (current) batch size and batch index computed with batch-pruning as we are looking into the pruned tensor
        uint64_t flattenedLogitIndex = beamHypIdx * (2*currentDimBatch*vocabSize) + 0 * (currentDimBatch*vocabSize) + currentBatchIdx*vocabSize + firstWordIdx;
        uint64_t secondFlattenedLogitIndex = beamHypIdx * (2*currentDimBatch*vocabSize) + 1 * (currentDimBatch*vocabSize) + currentBatchIdx*vocabSize + secondWordIdx;;  // (beam idx, batch idx, word idx); note: beam and batch are transposed, compared to 'key'
        // @TODO: use a function on shape() to index, or new method val->at({i1, i2, i3, i4}) with broadcasting
        ABORT_IF(lval->shape() != Shape({(int)nBestBeamSize, 1, (int)currentDimBatch, (int)vocabSize}) &&
                 (beamHypIdx == 0 && lval->shape() != Shape({1, 1, (int)currentDimBatch, (int)vocabSize})),
                 "Unexpected shape of logits?? {} != {}", lval->shape(), Shape({(int)nBestBeamSize, 1, (int)currentDimBatch, (int)vocabSize}));
        flattenedLogitIndices[j].push_back(flattenedLogitIndex);
        flattenedLogitIndices[j].push_back(secondFlattenedLogitIndex);
      }
      newBeamHypIndices.push_back(newBeam.size());
      origBatchIndices.push_back(origBatchIdx);
      oldBeamHypIndices.push_back(beamHypIdx);
    }

    // Set alignments
    if(!align.empty())
      hyp->setAlignment(getAlignmentsForHypothesis(align, batch, (int)beamHypIdx, (int)currentBatchIdx, (int)origBatchIdx, (int)currentDimBatch));
    else // not first factor: just copy
      hyp->setAlignment(beam[beamHypIdx]->getAlignment());

    newBeam.push_back(hyp);
  }

  // We need to set the score breakdown outside of the main loop to batch requests. This avoids issuing several 4 byte memcpys when using the GPU backend.
  if(options_->get<bool>("n-best")) {
    Tensor indices;
    Tensor logitsTensor;
    allocator_->allocate(indices, {(int)flattenedLogitIndices[0].size()}, Type::uint64);
    allocator_->allocate(logitsTensor, indices->shape(), Type::float32);
    std::vector<float> logits(flattenedLogitIndices[0].size());

    for(size_t state = 0; state < states.size(); ++state) {
      auto lval = states[state]->getLogProbs().getFactoredLogitsTensor(factorGroup); // [maxBeamSize, 2, currentDimBatch, dimFactorVocab]
      indices->set(flattenedLogitIndices[state]); 
      lval->gatherFromIndices(logitsTensor, indices);
      logitsTensor->get(logits);

      for(int i = 0; i < flattenedLogitIndices[state].size(); ++i) {
        const auto originalBatchIdx = origBatchIndices[i/2];
        const auto beamHypIdx = oldBeamHypIndices[i/2];
        const auto& beam = beams[originalBatchIdx];
        auto& newBeam = newBeams[originalBatchIdx];

        auto breakDown = beam[beamHypIdx]->getScoreBreakdown(); 
        breakDown.resize(states.size(), 0); // at start, this is empty, so this will set the initial score to 0
        breakDown[state] += logits[i];
        newBeam[newBeamHypIndices[i/2]]->setScoreBreakdown(breakDown);
      }
    }
    allocator_->free(indices);
    allocator_->free(logitsTensor);
  }

  // if factored vocab and this is not the first factor, we need to
  // also propagate factored hypotheses that do not get expanded in this step because they don't have this factor
  if (factorGroup > 0) {
    ABORT("not supported! note IBDecoder doesn't support factored models!");
    for (size_t batchIdx = 0; batchIdx < beams.size(); batchIdx++) {
      const auto& beam = beams[batchIdx];
      auto& newBeam = newBeams[batchIdx];
      for (const auto& beamHyp : beam) {
        auto word = beamHyp->getWord();
        //LOG(info, "Checking {}", factoredVocab->word2string(word));
        if (factoredVocab->canExpandFactoredWord(word, factorGroup)) // handled above
          continue;
        //LOG(info, "Forwarded {}", factoredVocab->word2string(word));
        newBeam.push_back(beamHyp);
      }
      if (newBeam.size() > beam.size()) {
        //LOG(info, "Size {}, sorting...", newBeam.size());
        std::nth_element(newBeam.begin(), newBeam.begin() + beam.size(), newBeam.end(), [](Hypothesis::PtrType a, Hypothesis::PtrType b) {
          return a->getPathScore() > b->getPathScore(); // (sort highest score first)
        });
        //LOG(info, "Size {}, sorted...", newBeam.size());
        newBeam.resize(beam.size());
      }
    }
  }
  return newBeams;
}

std::vector<float> BeamSearch::getAlignmentsForHypothesis( // -> P(s|t) for current t and given beam and batch dim
    const std::vector<float> alignAll, // [beam depth, max src length, batch size, 1], flattened vector of all attention probablities
    Ptr<data::CorpusBatch> batch,
    int beamHypIdx,
    int currentBatchIdx,
    int origBatchIdx,
    int currentDimBatch) const {
  // Let's B be the beam size, N be the number of batched sentences,
  // and L the number of words in the longest sentence in the batch.
  // The alignment vector:
  //
  // if(first)
  //   * has length of N x L if it's the first beam
  //   * stores elements in the following order:
  //     beam1 = [word1-batch1, word1-batch2, ..., word2-batch1, ...]
  // else
  //   * has length of N x L x B
  //   * stores elements in the following order:
  //     beams = [beam1, beam2, ..., beam_n]
  //
  // The mask vector is always of length N x L and has 1/0s stored like
  // in a single beam, i.e.:
  //   * [word1-batch1, word1-batch2, ..., word2-batch1, ...]
  //

  size_t origDimBatch = batch->size();  // number of sentences in batch
  size_t batchWidth   = batch->width(); // max src length

  // loop over words of batch entry 'currentBatchIdx' and beam entry 'beamHypIdx'
  std::vector<float> align;
  for(size_t srcPos = 0; srcPos < batchWidth; ++srcPos) { // loop over source positions
    // We are looking into the probabilites from an actual tensor, hence we need to use currentDimBatch and currentBatchIdx.
    size_t currentAttIdx = (batchWidth * beamHypIdx + srcPos) * currentDimBatch + currentBatchIdx; // = flatten [beam index, s, batch index, 0]

    // We are looking into the mask from the orginal batch, hence we need to use origDmBatch and origBatchIdx.
    size_t origAttIdx  = (batchWidth * beamHypIdx + srcPos) * origDimBatch + origBatchIdx;; // = flatten [beam index, s, batch index, 0]
    size_t origMaskIdx = origAttIdx % (batchWidth * origDimBatch); // == batchIdx + (batchSize * srcPos) = flatten [0, s, batch index, 0]

    // If the original position is not masked out used the corresponding current attention score.
    if(batch->front()->mask()[origMaskIdx] != 0)
      align.emplace_back(alignAll[currentAttIdx]);
  }
  return align;
}

// remove all beam entries that have reached EOS
Beams BeamSearch::purgeBeams(const Beams& beams, /*in/out=*/std::vector<IndexType>& batchIdxMap) {
  const auto trgEosId = trgVocab_->getEosId();
  Beams newBeams;
  size_t beamIdx = 0; // beam index
  for(auto beam : beams) {
    Beam newBeam; // a beam of surviving hyps
    for(auto hyp : beam)
      if(hyp->getSecondWord() != trgEosId && hyp->getWord() != trgEosId) // if this hyp is not finished,
        newBeam.push_back(hyp);      // move over to beam of surviving hyps

    if(PURGE_BATCH)
      if(newBeam.empty() && !beam.empty()) {      // previous beam had hyps, but all were finished in this step, newBeam will now stay empty
        for(size_t i = beamIdx + 1; i < beams.size(); ++i) // for all entries above this beam
          batchIdxMap[i] = batchIdxMap[i] - 1;  // make them look at one batch index below, as the current entry will be removed from the batch.
    }

    newBeams.push_back(newBeam);
    beamIdx++; // move to next beam index
  }
  return newBeams;
}

//**********************************************************************
// main decoding function
Histories BeamSearch::search(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch) {
  auto factoredVocab = trgVocab_->tryAs<FactoredVocab>();
  size_t numFactorGroups = factoredVocab ? factoredVocab->getNumGroups() : 1;
  if (numFactorGroups == 1) // if no factors then we didn't need this object in the first place
    factoredVocab.reset();

  // We will use the prefix "origBatch..." whenever we refer to batch dimensions of the original batch. These do not change during search.
  // We will use the prefix "currentBatch.." whenever we refer to batch dimension that can change due to batch-pruning.
  const int origDimBatch = (int)batch->size();
  const auto trgEosId = trgVocab_->getEosId();
  const auto trgUnkId = trgVocab_->getUnkId();

  auto getNBestList = createGetNBestListFn(beamSize_, origDimBatch*2*beamSize_, graph->getDeviceId());
  allocator_ = graph->getTensorAllocator();

  for(auto scorer : scorers_) {
    scorer->clear(graph);
  }

  Histories histories(origDimBatch);
  for(int i = 0; i < origDimBatch; ++i) {
    size_t sentId = batch->getSentenceIds()[i];
    histories[i] = New<History>(sentId,
                                trgEosId,
                                options_->get<float>("normalize"),
                                options_->get<float>("word-penalty"));
  }

  // start states
  std::vector<Ptr<ScorerState>> states;
  for(auto scorer : scorers_) {
    states.push_back(scorer->startState(graph, batch));
  }

  // create one beam per batch entry with sentence-start hypothesis
  Beams beams(origDimBatch, Beam(beamSize_, Hypothesis::New())); // array [origDimBatch] of array [maxBeamSize] of Hypothesis, keeps full size through search.
                                                                 // batch purging is determined from an empty sub-beam.
  std::vector<IndexType> batchIdxMap(origDimBatch); // Record at which batch entry a beam is looking.
                                                    // By default that corresponds to position in array,
                                                    // but shifts in the course of removing batch entries when they are finished.

  const std::vector<bool> emptyBatchEntries; // used for recording if there are empty input batch entries
  for(int origBatchIdx = 0; origBatchIdx < origDimBatch; ++origBatchIdx) {
    batchIdxMap[origBatchIdx] = origBatchIdx; // map to same position on initialization
    auto& beam = beams[origBatchIdx];
    histories[origBatchIdx]->add(beam, trgEosId); // add beams with start-hypotheses to traceback grid

    // Mark batch entries that consist only of source <EOS> i.e. these are empty lines. They will be forced to EOS and purged from batch
    const auto& srcEosId = batch->front()->vocab()->getEosId();
    const_cast<std::vector<bool>&>(emptyBatchEntries).push_back(batch->front()->data()[origBatchIdx] == srcEosId); // const_cast during construction
  }

  // determine index of UNK in the log prob vectors if we want to suppress it in the decoding process
  int unkColId = -1;
  std::vector<IndexType> unkOneHot(trgVocab_->size(), 0);
  if (trgUnkId != Word::NONE && !options_->get<bool>("allow-unk", false)) { // do we need to suppress unk?
    unkColId = factoredVocab ? factoredVocab->getUnkIndex() : trgUnkId.toWordIndex(); // what's the raw index of unk in the log prob vector?
    auto shortlist = scorers_[0]->getShortlist();      // first shortlist is generally ok, @TODO: make sure they are the same across scorers?
    if (shortlist)
      unkColId = shortlist->tryForwardMap(unkColId); // use shifted postion of unk in case of using a shortlist, shortlist may have removed unk which results in -1
    unkOneHot[unkColId] = 1;
  }
  auto unkOneHotTensor = graph->constant({1, 1, 1, (int) unkOneHot.size()}, inits::fromVector(unkOneHot));

  // the decoding process updates the following state information in each output time step:
  //  - beams: array [origDimBatch] of array [maxBeamSize] of Hypothesis
  //     - current output time step's set of active hypotheses, aka active search space
  //  - states[.]: ScorerState
  //     - NN state; one per scorer, e.g. 2 for ensemble of 2
  // and it forms the following return value
  //  - histories: array [origDimBatch] of History
  //    with History: vector [t] of array [maxBeamSize] of Hypothesis
  //    with Hypothesis: (last word, aggregate score, prev Hypothesis)

  IndexType currentDimBatch = origDimBatch;
  auto prevBatchIdxMap = batchIdxMap; // [origBatchIdx -> currentBatchIdx] but shifted by one time step
  Expr expandedScores;
  // main loop over output time steps
  for (size_t t = 0; ; t++) {
    ABORT_IF(origDimBatch != beams.size(), "Lost a batch entry??");
    // determine beam size for next output time step, as max over still-active sentences
    // E.g. if all batch entries are down from beam 5 to no more than 4 surviving hyps, then
    // switch to beam of 4 for all. If all are done, then beam ends up being 0, and we are done.
    size_t maxBeamSize = 0; // @TODO: is there some std::algorithm for this?
    for(auto& beam : beams)
      if(beam.size() > maxBeamSize)
        maxBeamSize = beam.size();

    // done if all batch entries have reached EOS on all beam entries
    if (maxBeamSize == 0)
      break;

    for (size_t factorGroup = 0; factorGroup < numFactorGroups; factorGroup++) {
      // for factored vocabs, we do one factor at a time, but without updating the scorer for secondary factors

      //**********************************************************************
      // create constant containing previous path scores for current beam
      // Also create mapping of hyp indices, for reordering the decoder-state tensors.
      std::vector<IndexType> batchIndices;    // [1,           1, currentDimBatch, 1] indices of currently used batch indices with regard to current, actual tensors
      std::vector<IndexType> hypIndices;      // [maxBeamSize, 1, currentDimBatch, 1] (flattened) tensor index ((beamHypIdx, batchIdx), flattened) of prev hyp that a hyp originated from
      std::vector<Word> prevWords;            // [maxBeamSize, 1, currentDimBatch, 1] (flattened) word that a hyp ended in, for advancing the decoder-model's history
      Expr prevPathScores;                    // [maxBeamSize, 1, currentDimBatch, 1], path score that a hyp ended in (last axis will broadcast into vocab size when adding expandedPathScores)

      bool anyCanExpand = false; // stays false if all hyps are invalid factor expansions
      if(t == 0 && factorGroup == 0) { // no scores yet
        prevPathScores = graph->constant({1, 1, 1, 1}, inits::fromValue(0));
        anyCanExpand = true;

        // at the beginning all batch entries are used
        batchIndices.resize(origDimBatch);
        std::iota(batchIndices.begin(), batchIndices.end(), 0);
      } else {
        if(factorGroup == 0)                                                              // only factorGroup==0 can subselect neural state
          for(int currentBatchIdx = 0; currentBatchIdx < beams.size(); ++currentBatchIdx) // loop over batch entries (active sentences)
            if(!beams[currentBatchIdx].empty() || !PURGE_BATCH)                           // for each beam check
              batchIndices.push_back(prevBatchIdxMap[currentBatchIdx]);                   // which batch entries were active in previous step

        std::vector<float> prevScores;
        for(size_t beamHypIdx = 0; beamHypIdx < maxBeamSize; ++beamHypIdx) { // loop over globally maximal beam-size (maxBeamSize)
          for(int origBatchIdx = 0; origBatchIdx < origDimBatch; ++origBatchIdx) { // loop over all batch entries (active and inactive)
            auto& beam = beams[origBatchIdx];
            if(beamHypIdx < beam.size()) {
              auto hyp = beam[beamHypIdx];
              auto canExpand = (!factoredVocab || factoredVocab->canExpandFactoredWord(hyp->getSecondWord(), factorGroup));
              //LOG(info, "[{}, {}] Can expand {} with {} -> {}", batchIdx, beamHypIdx, (*batch->back()->vocab())[hyp->getWord()], factorGroup, canExpand);
              anyCanExpand |= canExpand;

              auto currentBatchIdx = origBatchIdx;
              if(PURGE_BATCH) {
                if(factorGroup == 0)
                  currentBatchIdx = prevBatchIdxMap[origBatchIdx]; // subselection may happen for factorGroup == 0
                else
                  currentBatchIdx = batchIdxMap[origBatchIdx];     // no subselection happens for factorGroup > 0,
                                                                   // but we treat it like a next step, since a step
                                                                   // happened for factorGroup == 0
              }

              auto hypIndex = (IndexType)(hyp->getPrevStateIndex() * currentDimBatch + currentBatchIdx); // (beamHypIdx, batchIdx), flattened, for index_select() operation

              hypIndices.push_back(hypIndex); // (beamHypIdx, batchIdx), flattened as said above.
              prevWords.push_back(hyp->getWord());
              prevWords .push_back(hyp->getSecondWord());
              prevScores.push_back(canExpand ? hyp->getPathScore() : INVALID_PATH_SCORE);
            } else {  // pad to maxBeamSize (dummy hypothesis)
              if(!PURGE_BATCH || !beam.empty()) { // but only if we are not pruning and the beam is not deactivated yet
                hypIndices.push_back(0);
                prevWords.push_back(trgEosId);    // (unused, but must be valid)
                prevWords.push_back(trgEosId);
                prevScores.push_back((float)INVALID_PATH_SCORE);
              }
            }
          }
        }
        if(factorGroup == 0)
          currentDimBatch = (IndexType) batchIndices.size(); // keep batch size constant for all factor groups in a time step
        // Avoid unnecessary memcpy on GPU  
        if(anyCanExpand) prevPathScores = graph->constant({(int)maxBeamSize, 1, (int)currentDimBatch, 1}, inits::fromVector(prevScores));
      }
      if (!anyCanExpand) // all words cannot expand this factor: skip
        continue;

      //**********************************************************************
      // compute expanded path scores with word prediction probs from all scorers
      //   auto expandedPathScores = prevPathScores; // will become [maxBeamSize, 1, currDimBatch, dimVocab]
      expandedScores = graph->constant({1, 1, 1, 1}, inits::fromValue(0));
      Expr logProbs;
      for(size_t i = 0; i < scorers_.size(); ++i) {
          if (factorGroup == 0) {
          // compute output probabilities for current output time step
          //  - uses hypIndices[index in beam, 1, batch index, 1] to reorder scorer state to reflect the top-N in beams[][]
          //  - adds prevWords [index in beam, 1, batch index, 1] to the scorer's target history
          //  - performs one step of the scorer
          //  - returns new NN state for use in next output time step
          //  - returns vector of prediction probabilities over output vocab via newState
          // update state in-place for next output time step
          //if (t > 0) for (size_t kk = 0; kk < prevWords.size(); kk++)
          //  LOG(info, "prevWords[{},{}]={} -> {}", t/numFactorGroups, factorGroup,
          //      factoredVocab ? factoredVocab->word2string(prevWords[kk]) : (*batch->back()->vocab())[prevWords[kk]],
          //      prevScores[kk]);
          states[i] = scorers_[i]->step(graph, states[i], hypIndices, prevWords, batchIndices, (int)maxBeamSize);
          if (numFactorGroups == 1) // @TODO: this branch can go away
              logProbs = states[i]->getLogProbs().getLogits(); // [maxBeamSize, 1, currentDimBatch, dimVocab]
          else
          {
              auto shortlist = scorers_[i]->getShortlist();
              logProbs = states[i]->getLogProbs().getFactoredLogits(factorGroup, shortlist); // [maxBeamSize, 1, currentDimBatch, dimVocab]
          }
          }
          else {
          // add secondary factors
          // For those, we don't update the decoder-model state in any way.
          // Instead, we just keep expanding with the factors.
          // We will have temporary Word entries in hyps with some factors set to FACTOR_NOT_SPECIFIED.
          // For some lemmas, a factor is not applicable. For those, the factor score is the same (zero)
          // for all factor values. This would thus unnecessarily pollute the beam with identical copies,
          // and push out other hypotheses. Hence, we exclude those here by setting the path score to
          // INVALID_PATH_SCORE. Instead, toHyps() explicitly propagates those hyps by simply copying the
          // previous hypothesis.
          logProbs = states[i]->getLogProbs().getFactoredLogits(factorGroup, /*shortlist=*/ nullptr, hypIndices, maxBeamSize); // [maxBeamSize, 1, currentDimBatch, dimVocab]
          }
          // expand all hypotheses, [maxBeamSize, 1, currentDimBatch, 1] -> [maxBeamSize, 2, currentDimBatch, dimVocab]
          expandedScores = expandedScores + scorers_[i]->getWeight() * logProbs;
      }
      // IBDecoder: there will be two words predicted outside
      expandedScores = swapAxes(expandedScores, 0, 2); // -> [currentDimBatch, 2, maxBeamSize, dimVocab]

      // // perform NN computation
      // if(t == 0 && factorGroup == 0)
      //   graph->forward();
      // else
      //   graph->forwardNext();

      //**********************************************************************
      // suppress specific symbols if not at right positions
      if(unkColId != -1 && factorGroup == 0) {
        // suppressWord(expandedScores, unkColId);
        // Change to a fancy way to do the suppression, [0 0 0 0 0 1] => float type generally
        auto unkSuppressionTensor = cast(unkOneHotTensor, expandedScores->value_type());
        // set all values for the unk position to -inf
        expandedScores = expandedScores + unkSuppressionTensor * NumericLimits<float>(expandedScores->value_type()).lowest;
      }
      for(auto state : states)
        state->blacklist(expandedScores, batch);

      // IBDecoder: extract topN predictions for each word
      auto epsShape = expandedScores->shape();
      // // Note here we first compress the prediction for each word independently to reduce the search space and 
      // // hopefully increase running efficiency
      // expandedScores = reshape(expandedScores, {epsShape[-4]*2*epsShape[-2], 1, 1, epsShape[-1]});
      size_t topN = maxBeamSize;
      size_t actualBeamSize = epsShape[-2], vocabSize = epsShape[-1];

      // std::vector<unsigned int> nBestLocalKeys; // [currentDimBatch, 2, maxBeamSize, topN] flattened -> (batchIdx*2, beamHypIdx, word idx) flattened
      // std::vector<float> nBestLocalScores;  // [currentDimBatch, 2, maxBeamSize, topN] flattened
      // getNBestList(/*in*/   expandedScores->val(),  // [currentDimBatch*2*actualBeamSize, 1, 1, dimVocab or dimShortlist]
      //              /*N=*/   topN,                   // topK prediction
      //              /*out*/  nBestLocalScores,
      //              /*out*/  nBestLocalKeys,
      //              /*first*/true                    // the beam size dimension is always 1 so set the first flag to true
      // );
      // auto nBestLocalTopNScores = graph->constant(
      //   {(int)currentDimBatch, 2, (int)actualBeamSize, (int)topN}, inits::fromVector(nBestLocalScores));

      // IBDecoder: extract TopN predictions for each word via top-K operator
      // expandedScores
      auto expandedScoresTopNTuple = topk(expandedScores, topN, -1);  // [currentDimBatch, 2, actualBeamSize, topN]
      auto nBestLocalTopNScores    = get<0>(expandedScoresTopNTuple);
      auto nBestLocalTopNKeys      = get<1>(expandedScoresTopNTuple);

      // [currentDimBatch, actualBeamSize, 1, topN]
      auto firstWordTopNScores  = swapAxes(index_select(nBestLocalTopNScores, 1, std::vector<IndexType>(1, 0)), 1, 2);
      auto secondWordTopNScores = swapAxes(index_select(nBestLocalTopNScores, 1, std::vector<IndexType>(1, 1)), 1, 2);
      // [currentDimBatch, actualBeamSize, TopN1, TopN2]
      auto localTopNScores      = swapAxes(firstWordTopNScores, 2, 3) + secondWordTopNScores;
      localTopNScores           = reshape(localTopNScores, {(int)(localTopNScores->shape().elements()/(topN*topN)), 1, 1, (int)(topN*topN)});

      // perform NN computation
      // if(t == 0 && factorGroup == 0)
      //   graph->forward();
      // else
      // graph->forwardNext();

      // std::vector<unsigned int> nBestPairKeys; // [currentDimBatch, maxBeamSize] flattened -> (batchIdx, word1 topn index, word2 topn idx) flattened
      // std::vector<float> nBestPairScores;  // [currentDimBatch, maxBeamSize] flattened
      // getNBestList(/*in*/   localTopNScores->val(),  // [currentDimBatch*actualBeamSize, 1, 1, topN*topN]
      //              /*N=*/   topN,                    // topK prediction
      //              /*out*/  nBestPairScores,
      //              /*out*/  nBestPairKeys,
      //              /*first*/true                    // the beam size dimension is always 1 so set the first flag to true
      // );
      // auto nBestPairTopNScores = graph->constant(
      //   {(int)currentDimBatch, 1, (int)actualBeamSize, (int)topN}, inits::fromVector(nBestPairScores));
      auto localTopNScoresTopNTuple   = topk(localTopNScores, topN, -1);  // [currentDimBatch, 1, actualBeamSize, topN]
      auto nBestPairTopNScores        = get<0>(localTopNScoresTopNTuple);
      auto nBestPairTopNKeys          = get<1>(localTopNScoresTopNTuple);
      nBestPairTopNScores             = reshape(nBestPairTopNScores, {(int)currentDimBatch, 1, (int)actualBeamSize, (int)topN});
      
      // make beams continuous
      auto expandedPathScores = swapAxes(prevPathScores, 0, 2); // -> [currentDimBatch, 1, maxBeamSize, topN]
      expandedPathScores = expandedPathScores + nBestPairTopNScores;

      // perform NN computation
      if(t == 0 && factorGroup == 0)
        graph->forward();
      else
        graph->forwardNext();

      //**********************************************************************
      // perform beam search

      // find N best amongst the (maxBeamSize * dimVocab) hypotheses
      std::vector<unsigned int> nBestKeys; // [currentDimBatch, maxBeamSize] flattened -> (batchIdx, beamHypIdx, word idx) flattened
      std::vector<float> nBestPathScores;  // [currentDimBatch, maxBeamSize] flattened
      getNBestList(/*in*/   expandedPathScores->val(),   // [currentDimBatch, 1, maxBeamSize, topN]
                  /*N=*/    maxBeamSize,                 // desired beam size
                  /*out*/   nBestPathScores,
                   /*out*/  nBestKeys,
                  /*first=*/t == 0 && factorGroup == 0); // @TODO: this is only used for checking presently, and should be removed altogether
      // Now, nBestPathScores contain N-best expandedPathScores for each batch and beam,
      // and nBestKeys for each their original location (batchIdx, beamHypIdx, word).

      // IBDecoder: extract data from GPU to CPU
      std::vector<IndexType> nBestPairKeys, nBestLocalKeys;
      nBestLocalTopNKeys->val()->get(nBestLocalKeys);
      nBestPairTopNKeys->val()->get(nBestPairKeys);

      // IBDecoder: update nBestKey to extract the vocabulary information
      std::vector<uint64_t> nNewBestKeys;
      for(size_t i = 0; i < nBestKeys.size(); ++i) { // [currentDimBatch, maxBeamSize] flattened
        const auto  key = nBestKeys[i];
        
        // decompose key into individual indices (batchIdx, beamHypIdx, wordIdx)
        const auto beamHypIdx      = (key / topN) % actualBeamSize;
        const auto currentBatchIdx = (key / topN) / actualBeamSize;
        // const auto predIdx         = (key % topN);

        // backtrace the pair information
        // [currentDimBatch*actualBeamSize, 1, 1, topN*topN]
        const auto pairKey = nBestPairKeys[key];
        const auto topNSquare = topN * topN;

        // decompose key into individual indices
        // const auto pairBeamHypIdx       = (pairKey / topNSquare) % actualBeamSize;
        // const auto pairCurrentBatchIdx  = (pairKey / topNSquare) / actualBeamSize;
        const auto pairPredIdx          = (pairKey % topNSquare);

        // ABORT_IF(beamHypIdx != pairBeamHypIdx, "Beam hypothesis index should be the same");
        // ABORT_IF(currentBatchIdx != pairCurrentBatchIdx, "Batch index should also be the same");

        // backtrace the actual word information
        const auto firstWordPredIdx     = (pairPredIdx / topN);
        const auto secondWordPredIdx    = (pairPredIdx % topN);

        const auto firstWordLocalIdx    = currentBatchIdx * (2*actualBeamSize*topN) + 0 * (actualBeamSize*topN) + beamHypIdx * (topN) + firstWordPredIdx;
        const auto secondWordLocalIdx   = currentBatchIdx * (2*actualBeamSize*topN) + 1 * (actualBeamSize*topN) + beamHypIdx * (topN) + secondWordPredIdx;

        // encode the word information
        // const auto firstWordKey  = nBestLocalKeys[firstWordLocalIdx ];
        // const auto secondWordKey = nBestLocalKeys[secondWordLocalIdx];  
        const auto firstWord     = nBestLocalKeys[firstWordLocalIdx ] % vocabSize;
        const auto secondWord    = nBestLocalKeys[secondWordLocalIdx] % vocabSize;

        // // {(int)currentDimBatch, 2, (int)actualBeamSize, (int)vocab}
        // const auto firstWordBeamHypIdx    = (firstWordKey  / vocabSize) % actualBeamSize;
        // const auto firstWordBatchIdx      = (firstWordKey  / vocabSize)  / actualBeamSize / 2;
        // const auto secondWordBeamHypIdx   = (secondWordKey / vocabSize) % actualBeamSize;
        // const auto secondWordBatchIdx     = (secondWordKey / vocabSize)  / actualBeamSize / 2;

        // ABORT_IF(beamHypIdx != firstWordBeamHypIdx, "First Word Beam hypothesis index should be the same");
        // ABORT_IF(currentBatchIdx != firstWordBatchIdx, "First word Batch index should also be the same");
        // ABORT_IF(beamHypIdx != secondWordBeamHypIdx, "Second Word Beam hypothesis index should be the same");
        // ABORT_IF(currentBatchIdx != secondWordBatchIdx, "Second word Batch index should also be the same");

        // [currentDimBatch, maxBeamSize, vocabSize, vocabSize]
        // nBestKeys[i] = currentBatchIdx * (actualBeamSize*vocabSize*vocabSize) + beamHypIdx * (vocabSize*vocabSize) + firstWord * (vocabSize) + secondWord;
        const uint64_t newKey = currentBatchIdx * (actualBeamSize*vocabSize*vocabSize) + beamHypIdx * (vocabSize*vocabSize) + firstWord * (vocabSize) + secondWord;
        nNewBestKeys.push_back(newKey);
      }

      // combine N-best sets with existing search space (beams) to updated search space
      beams = toHyps(nNewBestKeys, nBestPathScores,
                     /*nBestBeamSize*/actualBeamSize, // used for interpretation of keys
                     /*vocabSize=*/vocabSize,    // used for interpretation of keys
                     beams,
                     states,            // used for keeping track of per-ensemble-member path score
                     batch,             // only used for propagating alignment info
                     factoredVocab, factorGroup,
                     emptyBatchEntries, // [origDimBatch] - empty source batch entries are marked with true
                     batchIdxMap);      // used to create a reverse batch index map to recover original batch indices for this step
    } // END FOR factorGroup = 0 .. numFactorGroups-1

    prevBatchIdxMap = batchIdxMap; // save current batchIdx map to be used in next step; we are then going to look one step back

    // remove all hyps that end in EOS
    // The position of a hyp in the beam may change.
    // in/out = shifts the batch index map if a beam gets fully purged
    const auto purgedNewBeams = purgeBeams(beams, /*in/out=*/batchIdxMap);

    // add updated search space (beams) to our return value
    bool maxLengthReached = false;
    for(int batchIdx = 0; batchIdx < origDimBatch; ++batchIdx) {
      // if this batch entry has surviving hyps then add them to the traceback grid
      if(!beams[batchIdx].empty()) { // if the beam is not empty expand the history object associated with the beam
        if (histories[batchIdx]->size() >= options_->get<float>("max-length-factor") * batch->front()->batchWidth())
          maxLengthReached = true;
        histories[batchIdx]->add(beams[batchIdx], trgEosId, purgedNewBeams[batchIdx].empty() || maxLengthReached);
      }
    }
    if (maxLengthReached) // early exit if max length limit was reached
      break;

    // this is the search space for the next output time step
    beams = purgedNewBeams;
  } // end of main loop over output time steps

  return histories; // [origDimBatch][t][N best hyps]
}

}  // namespace marian