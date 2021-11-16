#pragma once

#include "common/options.h"
#include "functional/functional.h"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "layers/factory.h"
#include "tensors/backend.h"
#include "tensors/tensor.h"
#include "tensors/tensor_allocator.h"
#include "tensors/tensor_operators.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace marian {

class IRegulariser {
protected:
  Ptr<ExpressionGraph> graph_;
  Ptr<Options> options_;

  float lambda_{0.0f};
  std::string type_{""};

  std::map<std::string, Expr> partialPenalties_; // to gather penalties for all layers
  std::map<std::string, Expr> masks_; // additional binary masks if needed somewhere

public:
  IRegulariser(Ptr<ExpressionGraph> graph, Ptr<Options> options, float lambda, std::string type)
      : graph_(graph), options_(options), lambda_(lambda), type_(type) {}
  IRegulariser() {}

  virtual ~IRegulariser() {}

  virtual std::map<std::string, float> updateStats(Ptr<ExpressionGraph> graph, float gradNorm) { return {  }; }

  virtual void synchroStats(std::map<std::string, float> scalars) {}

  virtual float getLambda() { return lambda_; }

  virtual std::string getType() { return type_; }

  virtual Expr getTotalPenalty() {
    Expr totalPenalty;
    // LOG(info, "IRegulariser getTotalPenalty wtf");
    for(const auto& partialPenalty : partialPenalties_) {
      if(!totalPenalty)
        totalPenalty = partialPenalty.second;
      else
        totalPenalty = totalPenalty + partialPenalty.second;
    }
    return lambda_ * totalPenalty;
  }

  virtual std::map<std::string, Expr> getPartialPenalties() { return partialPenalties_; }

  virtual void clear() { partialPenalties_.clear(); }

  // virtual Expr getMask( return nullptr; )

  virtual Expr calculatePenalty(Expr W, Expr b, bool rows = false, bool inference = false) = 0;
};


class IAidedRegulariser : public virtual IRegulariser {
protected:
  std::vector<Ptr<TensorAllocator>> allocators_;
  std::map<std::string, float> scalars_; // Each penalty will be scaled independently
  std::map<std::string, bool> flipped_; // Is it rows or columns

public:
  IAidedRegulariser(Ptr<ExpressionGraph> graph, Ptr<Options> options, float lambda, std::string type)
      : IRegulariser(graph, options, lambda, type) {}
  IAidedRegulariser() : IRegulariser() {}

  virtual ~IAidedRegulariser() {}

  void synchroStats(std::map<std::string, float> scalars) override {
    // LOG(info, "SynchroStats");
    scalars_ = scalars;
  }

  // void clear() override { partialPenalties_.clear(); }

};


class L0Regulariser : public IRegulariser {
protected:
  float gamma_{-0.1};
  float zeta_{1.1};
  float gamma_zeta_ratio_{std::log(-gamma_ / zeta_)};

public:
  L0Regulariser(Ptr<ExpressionGraph> graph, Ptr<Options> options, float lambda, std::string type)
      : IRegulariser(graph, options, lambda, type) {
        if (!options_)
          LOG(info, "options_ is nullptr???"); 
        
        if (!graph_)
          LOG(info, "graph_ is nullptr???");
        
        if (!options)
          LOG(info, "options coming in is nullptr???"); 
        
        if (!graph)
          LOG(info, "graph coming in is nullptr???");
      
      }

  Expr hard_sigmoid(Expr input, float low = 0.0, float high = 1.0) {
    return minimum(maximum(input, low), high); 
  }

  // L0-regularisation
  // non-differentiable so we need hard concrete distribution trick
  // we ignore bias in this case
  Expr coeffL0Penalty(Expr W, Expr b, bool rows = false, bool inference = false) {
    Expr s;
    auto loc = W->graph()->param(W->name() + "_l0_loc", W->shape(), inits::normal(0, 0.01));
    
    auto temp = 0.66;
    
    auto noise = W->graph()->constant(W->shape(), inits::uniform());

    // LOG(info, "inside L0 regularisating layer {} {}", W->name(), W->shape());
    if (!inference) {
      auto p = sum(sum(hard_sigmoid(sigmoid(loc - temp * gamma_zeta_ratio_), 1e-5, 1 - 1e-5) * abs(W), -1), -2);
      partialPenalties_.emplace(W->name(), p);
      
      s = sigmoid((log(noise) - log(1 - noise) + loc) / temp);
      s = s * (zeta_ - gamma_) + gamma_ * (p / p);  // trick to connect to graph
        
    }
    else {
      s = sigmoid(loc) * (zeta_ - gamma_) + gamma_;
    }
    
    return hard_sigmoid(s);
  }

  // Group L0
  // Basically, check what layer it is and based on it, either do head or rowcol
  Expr groupL0Penalty(Expr W, Expr b, bool rows = false, bool inference = false) {

    Shape newShape;
    Expr s;

    Expr WL2; 

    bool isAtt = W->name().find("self") != std::string::npos || W->name().find("context") != std::string::npos;
    bool isFFN = W->name().find("ffn") != std::string::npos; 
    
    if (isAtt) {
      int h = W->shape()[0];

      int blockH = W->shape()[0];                               // inner dimension = 256
      int blockW = options_->get<int>("transformer-head-dim");  // head size = 32

      int innerShape = W->shape()[0] * W->shape()[1] / (blockW * h);
      int blockNum   = W->shape()[0] * W->shape()[1] / (blockH * blockW);

      // splitting a matrix into separate heads
      // TODO: modify transformer model to split parameters first?
      // but could be inefficient matrix multiplication-wise

      auto reshaped = reshape(W, {h / blockH, blockH, innerShape, blockW});
      auto heads    = reshape(transpose(reshaped, {0, 2, 1, 3}), {1, blockNum, blockH, blockW});

      WL2 = sum(sum(heads * heads, -2), -1);
      if(!rows) {
        auto bBlocks = reshape(b, {b->shape()[1] / blockW, 1, blockW});
        auto bSum    = sum(bBlocks * bBlocks, -1);
        WL2         = WL2 + bSum;
      }

      newShape = {blockNum, 1, 1};
      WL2 = reshape(WL2, newShape);
      // debug(WL2, "WL2 in att");
    }

    else if (isFFN) {
      if (!rows) {
        WL2 = sum(W * W, -1);
        newShape = {1, W->shape()[1]};
      }
      else { 
        WL2 = sum(W * W, -2);
        newShape = {W->shape()[0], 1}; 
      }
      // debug(WL2, "WL2 in ffn");
    }
    
      
    auto loc = W->graph()->param(W->name() + "_l0_loc", newShape, inits::normal(0, 0.01));
    auto temp = 0.66;
    auto noise = W->graph()->constant(newShape, inits::uniform());

    if (!inference) {

      auto pOpen = sigmoid(loc - temp * gamma_zeta_ratio_);
      auto p = 0.5 * pOpen * WL2;
      // debug(pOpen, "pOpen");
      // debug(p, "penalty");
      // LOG(info, "inside L0 regularisating layer {} {} {}", W->name(), WL2->shape(), pOpen->shape());
      
      for (int i = 0; i < p->shape().size(); i++)
        p = sum(p, i);
      partialPenalties_.emplace(W->name(), p);
      
      s = sigmoid((log(noise) - log(1 - noise) + loc) / temp);
      s = s * (zeta_ - gamma_) + gamma_ * (p / p);  // trick to connect to graph
        
    }
    else {
      s = sigmoid(loc) * (zeta_ - gamma_) + gamma_;
    }

    auto mask = hard_sigmoid(s);

    return mask;
  }

  Expr calculatePenalty(Expr W, Expr b, bool rows = false, bool inference = false) override {
    Expr mask;
    if(type_ == "l0") {
      mask = coeffL0Penalty(W, b, rows);
    } else if(type_ == "l0-group") {
      mask = groupL0Penalty(W, b, rows);
    }
    return mask;
  }

};

class LhalfRegulariser : public IRegulariser {
public:
  LhalfRegulariser(Ptr<ExpressionGraph> graph, Ptr<Options> options, float lambda, std::string type)
      : IRegulariser(graph, options, lambda, type) {}

  // L0.5-regularisation
  // so since it is p = 0.5, parameters are sqrt and then added together with a square
  // we ignore bias in this case
  Expr calculatePenalty(Expr W, Expr b, bool rows = false, bool inference = false) override {
    auto p = square(sum(sum(sqrt(abs(W)), -1), -2));

    partialPenalties_.emplace(W->name(), p);
    return p;
  }
};

class L1Regulariser : public IRegulariser {

public:
  L1Regulariser(Ptr<ExpressionGraph> graph, Ptr<Options> options, float lambda, std::string type)
      : IRegulariser(graph, options, lambda, type) {}

  // L1-regularisation
  // just a sum of all absolute values
  // we ignore bias in this case
  Expr calculatePenalty(Expr W, Expr b, bool rows = false, bool inference = false) override {
    auto p = sum(sum(abs(W), -1), -2);

    partialPenalties_.emplace(W->name(), p);
    return p;
  }
};

class L2Regulariser : public IRegulariser {

public:
  L2Regulariser(Ptr<ExpressionGraph> graph, Ptr<Options> options, float lambda, std::string type)
      : IRegulariser(graph, options, lambda, type) {}

  // L2-regularisation
  // just a sum of square values
  // we ignore bias in this case
  Expr calculatePenalty(Expr W, Expr b, bool rows = false, bool inference = false) override {
    auto p = sum(sum(W * W, -1), -2);

    partialPenalties_.emplace(W->name(), p);
    return p;
  }
};

class ElasticRegulariser : public IRegulariser {

public:
  ElasticRegulariser(Ptr<ExpressionGraph> graph,
                     Ptr<Options> options,
                     float lambda,
                     std::string type)
      : IRegulariser(graph, options, lambda, type) {}

  // Elastic net regularisation
  // which is just L1 + L2
  // we ignore bias in this case
  Expr calculatePenalty(Expr W, Expr b, bool rows = false, bool inference = false) override {
    auto p1 = sum(sum(abs(W), -1), -2);
    auto p2 = sum(sum(W * W, -1), -2);
    auto p  = p1 + p2;

    partialPenalties_.emplace(W->name(), p);
    return p;
  }
};

class GroupLassoRegulariser : public virtual IRegulariser {
public:
  GroupLassoRegulariser(Ptr<ExpressionGraph> graph,
                        Ptr<Options> options,
                        float lambda,
                        std::string type)
      : IRegulariser(graph, options, lambda, type) {}
  GroupLassoRegulariser() : IRegulariser() {}

  Expr calculatePenalty(Expr W, Expr b, bool rows = false, bool inference = false) override {
    Expr p;
    if(type_ == "rowcol") {
      p = rowcolPenalty(W, b, rows);
    } else if(type_ == "heads") {
      p = headPenalty(W, b, rows);
    } else if(type_ == "rowcol-root") {
      p = rowcolRootPenalty(W, b, rows);
    } else if(type_ == "layer") {
      p = layerPenalty(W, b, rows);
    } else { // default to rowcol
      p = rowcolPenalty(W, b, rows);
    }

    partialPenalties_.emplace(W->name(), p);
    return p;
  }

protected:
  Expr rowcolRootPenalty(Expr W, Expr b, bool rows = false, bool inference = false) {
    size_t axisL2, axisL1;

    // depending on whether we regularise rows or columns, apply L1 and L2
    // alongside specific axes

    if(!rows) {
      axisL2 = -2;
      axisL1 = -1;
    } else {
      axisL2 = -1;
      axisL1 = -2;
    }

    auto WSum = sum(sqrt(abs(W)), axisL2);

    // if regularising columns, we also need to remove biases with L2
    if(!rows) {
      WSum = WSum + sqrt(abs(b));
    }

    auto p = sum(square(WSum), axisL1);

    auto scale = std::sqrt(W->shape()[0]);
    return scale * p;
  }

  Expr layerPenalty(Expr W, Expr b, bool rows = false, bool inference = false) {
    size_t axisL2, axisL1;

    // depending on whether we regularise rows or columns, apply L1 and L2
    // alongside specific axes

    if(!rows) {
      axisL2 = -2;
      axisL1 = -1;
    } else {
      axisL2 = -1;
      axisL1 = -2;
    }

    auto WSum = sum(W * W, axisL2);

    if(!rows)
      WSum = WSum + (b * b);

    auto p = sqrt(sum(WSum, axisL1));

    auto scale = std::sqrt(W->shape()[0]);
    return scale * p;
  }

  Expr rowcolPenalty(Expr W, Expr b, bool rows = false, bool inference = false) {
    size_t axisL2, axisL1;

    // depending on whether we regularise rows or columns, apply L1 and L2
    // alongside specific axes

    if(!rows) {
      axisL2 = -2;
      axisL1 = -1;
    } else {
      axisL2 = -1;
      axisL1 = -2;
    }

    // calculate mask, used in aided regulariser
    
    auto WMask = gt(sum(W, axisL2), 1e-5);
    masks_.emplace(W->name(), WMask);
    //

    auto WSum = sum(W * W, axisL2) * WMask;

    // if regularising columns, we also need to remove biases with L2
    if(!rows) {
      WSum = WSum + (b * b);
    }

    auto p = sum(sqrt(WSum), axisL1);

    float scale = 1.0f;
    // if(!rows)
      // scale = std::sqrt(W->shape()[0]);
    // else
      // scale = std::sqrt(W->shape()[1]);

    return scale * p;
  }

  Expr headPenalty(Expr W, Expr b, bool rows = false, bool inference = false) {
    int h = W->shape()[0];

    int blockH = W->shape()[0];                               // inner dimension = 256
    int blockW = options_->get<int>("transformer-head-dim");  // head size = 32

    int innerShape = W->shape()[0] * W->shape()[1] / (blockW * h);
    int blockNum   = W->shape()[0] * W->shape()[1] / (blockH * blockW);

    // splitting a matrix into separate heads
    // TODO: modify transformer model to split parameters first?
    // but could be inefficient matrix multiplication-wise

    auto reshaped = reshape(W, {h / blockH, blockH, innerShape, blockW});
    auto heads    = reshape(transpose(reshaped, {0, 2, 1, 3}), {1, blockNum, blockH, blockW});

    auto WSum = sum(sum(heads * heads, -2), -1);

    if(!rows) {
      auto bBlocks = reshape(b, {b->shape()[1] / blockW, 1, blockW});
      auto bSum    = sum(bBlocks * bBlocks, -1);
      WSum         = WSum + bSum;
    }

    // sum across all heads too
    float scale = 1.0f;
    auto p = sum(sqrt(WSum), -3);

    // I believe it's called orthonormalisation???
    // auto scale = std::sqrt(blockH * blockW);
    return scale * p;
  }
};


class AidedGroupLassoRegulariser : public IAidedRegulariser, public GroupLassoRegulariser {
protected:
  std::vector<Ptr<TensorAllocator>> allocators_;
  
  Tensor tempAvg_; // temp variable to hold the new average
  float alpha_{0.0001}; // alpha for exponential moving average of gradients

  bool isInitialised_{false};

public:
  AidedGroupLassoRegulariser(Ptr<ExpressionGraph> graph, Ptr<Options> options, float lambda, std::string type)
      : IRegulariser(graph, options, lambda, type) { LOG(info, "AidedGroupLasso constructor");}

  using IAidedRegulariser::synchroStats;
  using IAidedRegulariser::updateStats;

  virtual ~AidedGroupLassoRegulariser() {}
  
  std::map<std::string, float> updateStats(Ptr<ExpressionGraph> graph, float gradNorm) override {

    // LOG(info, "AidedGroupLasso UPDATE STAAAAAATS");

    if(!graph_) {
      LOG(info, "WHY IS GRAPH POINTER EMPTY HERE IN updateStats???");
      graph_ = graph;
    }

    if (!tempAvg_) {
        auto allocator = New<TensorAllocator>(graph_->getBackend());
        allocator->reserveExact(graph_->params()->vals()->memory()->size());
        allocator->allocate(tempAvg_, {1, 1});

        // tempAvg_->set(0);

        allocators_.push_back(allocator);
    }

    for (const auto &s : scalars_) { // for all layers that we regularise
      auto name = s.first;
      auto node = graph_->get(name);

      if (!node)
        LOG(info, "node of the name {} is nullptr???", name);

      // Average the abs of gradients for the layer
      
      // float dim = node->shape()[1];
      // if (flipped_[name])
        // dim = node->shape()[0];

      using namespace functional;
      // Reduce(abs(_1), 1.0f / dim, tempAvg_, node->grad());
      Reduce(_1 * _1, 1.0f, tempAvg_, node->grad());
      // float newScalar = std::sqrt(tempAvg_->get(0)) * (1.0f / dim);
      float sqrtScalar = std::sqrt(tempAvg_->get(0));
      float newScalar = std::abs(std::log(sqrtScalar / gradNorm));
      tempAvg_->set(0);

      // float newScalar = 1.0f;

      if (scalars_[name] == 0.0f) { // if no previous statistics were done
        // LOG(info, "ZEROOOOOO");
        scalars_[name] = newScalar; 
      }
      else { // do exponential moving average if previous statistic exist
        scalars_[name] = alpha_ * newScalar + (1 - alpha_) * scalars_[name]; 
      }

      // LOG(info, "updatedStats, global gradNorm={}, node={}, gradScalar={}, sqrtScalar={}, scalar={} flipped={} dim={}", gradNorm, name, newScalar, sqrtScalar, scalars_[name], flipped_[name], dim); 
    }

    return scalars_;
  }
  
  Expr calculatePenalty(Expr W, Expr b, bool rows = false, bool inference = false) override {
    // if (scalars_.find(W->name()) == scalars_.end()) { // if scalar for the node doesn't yet exist 
      // scalars_[W->name()] = 1.0f;
    // }
    // auto p = GroupLassoRegulariser::calculatePenalty(W, b, rows, inference);
    
    // store whether it's rows or columns
    flipped_[W->name()] = rows;
    
    Expr p;
    if(type_ == "rowcol") {
      p = GroupLassoRegulariser::rowcolPenalty(W, b, rows);
    } else if(type_ == "heads") {
      p = GroupLassoRegulariser::headPenalty(W, b, rows);
    } else if(type_ == "rowcol-root") {
      p = GroupLassoRegulariser::rowcolRootPenalty(W, b, rows);
    } else if(type_ == "layer") {
      p = GroupLassoRegulariser::layerPenalty(W, b, rows);
    } else { // default to rowcol
      p = GroupLassoRegulariser::rowcolPenalty(W, b, rows);
    }
    // debug(p, "before scalar");
    // LOG(info, "calculatePenalty AidedGroupLasso, node={} scalar={}", W->name(), scalars_[W->name()]);
    // debug(p, W->name() + "NORMAL in calculatePenalty???");
    if (scalars_[W->name()] != 0.0f) {
      // LOG(info, "SCALING THE NODE {} {}", W->name(), p->shape());
      p = p * scalars_[W->name()];
    }
    // debug(p, "after scalar");
    partialPenalties_.emplace(W->name(), p);
    // debug(p, W->name() + " SCALED in calculatePenalty???");
    return p;
  }

};


class RegulariserFactory : public Factory {
public:
  using Factory::Factory;
  RegulariserFactory(Ptr<Options> options) : Factory(options){};

  Ptr<IRegulariser> construct(Ptr<ExpressionGraph> graph, float lambda, std::string type) {
    LOG_ONCE(info, "Regulariser type {}", type);
    if(type == "l0") {
      LOG_ONCE(info, "Regularisation type selected: l0, shape=coeff");
      return New<L0Regulariser>(graph, options_, lambda, type);
    }
    else if(type == "l0-group") {
      LOG_ONCE(info, "Regularisation type selected: l0, shape=group");
      return New<L0Regulariser>(graph, options_, lambda, type);
    }
    else if(type == "l1") {
      LOG_ONCE(info, "Regularisation type selected: l1");
      return New<L1Regulariser>(graph, options_, lambda, type);
    } else if(type == "l2") {
      LOG_ONCE(info, "Regularisation type selected: l2");
      return New<L2Regulariser>(graph, options_, lambda, type);
    } else if(type == "lhalf") {
      LOG_ONCE(info, "Regularisation type selected: lhalf");
      return New<LhalfRegulariser>(graph, options_, lambda, type);
    } else if(type == "elastic") {
      LOG_ONCE(info, "Regularisation type selected: elastic");
      return New<ElasticRegulariser>(graph, options_, lambda, type);
    } else if(type == "rowcol") {
      LOG_ONCE(info, "Regularisation type selected: group lasso, shape=rowcol");
      return New<GroupLassoRegulariser>(graph, options_, lambda, type);
    } else if(type == "rowcol-root") {
      LOG_ONCE(info, "Regularisation type selected: group lasso, shape=rowcol-root");
      return New<GroupLassoRegulariser>(graph, options_, lambda, type);
    } else if(type == "layer") {
      LOG_ONCE(info, "Regularisation type selected: group lasso, shape=layer");
      return New<GroupLassoRegulariser>(graph, options_, lambda, type);
    } else if(type == "heads") {
      LOG_ONCE(info, "Regularisation type selected: group lasso, shape=heads");
      return New<GroupLassoRegulariser>(graph, options_, lambda, type);
    } else if(type == "aided") {
      LOG_ONCE(info, "Regularisation type selected: aided group lasso, shape=rowcol");
      return New<AidedGroupLassoRegulariser>(graph, options_, lambda, type);
    } else {
      LOG_ONCE(
          info,
          "Regularisation type selected but not on the list? Returning nullptr, will break? {}",
          type);
      return nullptr;
    }
  }
};
}
