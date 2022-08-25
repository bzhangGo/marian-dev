#pragma once

#include "marian.h"
#include "states.h"

#include "data/shortlist.h"
#include "layers/constructors.h"
#include "layers/generic.h"

namespace marian {

class DecoderBase : public EncoderDecoderLayerBase {
protected:
  Ptr<data::Shortlist> shortlist_;

public:
  DecoderBase(Ptr<ExpressionGraph> graph, Ptr<Options> options) :
    EncoderDecoderLayerBase(graph, options, "decoder", /*batchIndex=*/1,
        options->get<float>("dropout-trg", 0.0f),
        options->get<bool>("embedding-fix-trg", false)) {}

  virtual Ptr<DecoderState> startState(Ptr<ExpressionGraph>,
                                       Ptr<data::CorpusBatch> batch,
                                       std::vector<Ptr<EncoderState>>&)
      = 0;

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph>, Ptr<DecoderState>) = 0;

  virtual void embeddingsFromBatch(Ptr<ExpressionGraph> graph,
                                   Ptr<DecoderState> state,
                                   Ptr<data::CorpusBatch> batch) {
    graph_ = graph;

    auto subBatch = (*batch)[batchIndex_];

    Expr y, yMask; std::tie
    (y, yMask) = getEmbeddingLayer()->apply(subBatch);

    // @TODO: during training there is currently no code path that leads to using a shortlist
#if 0
    const Words& data =
      /*if*/ (shortlist_) ?
        shortlist_->mappedIndices()
      /*else*/ :
        subBatch->data();
#endif

    ABORT_IF(shortlist_, "How did a shortlist make it into training?");

    // IBDecoder: shiting by 2 rather than 1
    auto yDelayed = shift(y, {2, 0, 0}); // insert zero at front; first word gets predicted from a target embedding of 0

    state->setTargetHistoryEmbeddings(yDelayed);
    state->setTargetMask(yMask);
    
    const Words& data = subBatch->data();
    state->setTargetWords(data);
  }

  virtual void embeddingsFromPrediction(Ptr<ExpressionGraph> graph,
                                        Ptr<DecoderState> state,
                                        const Words& words,
                                        int dimBatch,
                                        int dimBeam) {
    graph_ = graph;
    auto embeddingLayer = getEmbeddingLayer();
    Expr selectedEmbs;
    int dimEmb = opt<int>("dim-emb");
    // IBDecoder: one word prediction to two words prediction
    if(words.empty())
      selectedEmbs = graph_->constant({1, 2, dimBatch, dimEmb}, inits::zeros());
    else {
      selectedEmbs = embeddingLayer->apply(words, {dimBeam, dimBatch, 2, dimEmb});
      // IBDecoder: reorder the shaping
      selectedEmbs = swapAxes(selectedEmbs, 1, 2);  // [dimBeam, 2, dimBatch, dimEmb]
    }
    state->setTargetHistoryEmbeddings(selectedEmbs);
  }

  virtual const std::vector<Expr> getAlignments(int /*i*/ = 0) { return {}; }; // [tgt index][beam depth, max src length, batch size, 1]
  
  virtual std::vector<Ptr<IRegulariser>> getRegularisers(int /*i*/ = 0) { return {}; };

  virtual Ptr<data::Shortlist> getShortlist() { return shortlist_; }
  virtual void setShortlist(Ptr<data::Shortlist> shortlist) {
    shortlist_ = shortlist;
  }

  virtual void clear() = 0;
};

}  // namespace marian
