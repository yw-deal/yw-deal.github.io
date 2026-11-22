/*
 * mlp-core.js —— 从零实现的多层感知机（MLP）
 * 纯 JavaScript，无任何依赖：前向传播、反向传播、SGD / Momentum / Adam。
 * 同时支持浏览器（window.MLPCore）与 Node.js（module.exports），便于本地测试。
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.MLPCore = factory();
  }
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  /* ---------------- 随机数与初始化 ---------------- */

  function mulberry32(a) {
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(rand) {
    let u = 0;
    let v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const ACTIVATIONS = {
    tanh: {
      label: 'Tanh',
      init: 'xavier',
      f: function (z) {
        return Math.tanh(z);
      },
      df: function (z, a) {
        return 1 - a * a;
      },
    },
    relu: {
      label: 'ReLU',
      init: 'he',
      f: function (z) {
        return z > 0 ? z : 0;
      },
      df: function (z) {
        return z > 0 ? 1 : 0;
      },
    },
    sigmoid: {
      label: 'Sigmoid',
      init: 'xavier',
      f: function (z) {
        return 1 / (1 + Math.exp(-z));
      },
      df: function (z, a) {
        return a * (1 - a);
      },
    },
  };

  /* ---------------- MLP ---------------- */

  function MLP(sizes, options) {
    options = options || {};
    this.sizes = sizes.slice();
    this.L = sizes.length - 1;
    this.activationName = options.activation || 'tanh';
    this.act = ACTIVATIONS[this.activationName];
    if (!this.act) throw new Error('未知激活函数: ' + this.activationName);

    const rand = mulberry32(options.seed == null ? 42 : options.seed);
    this.W = [];
    this.b = [];

    for (let l = 0; l < this.L; l++) {
      const fanIn = sizes[l];
      const fanOut = sizes[l + 1];
      const isOutput = l === this.L - 1;
      const scale = this.act.init === 'he' ? Math.sqrt(2 / fanIn) : Math.sqrt(1 / fanIn);
      const layer = [];
      for (let j = 0; j < fanOut; j++) {
        const row = new Float64Array(fanIn);
        for (let k = 0; k < fanIn; k++) {
          row[k] = gaussian(rand) * scale * (isOutput ? 1 : 1);
        }
        layer.push(row);
      }
      this.W.push(layer);
      this.b.push(new Float64Array(fanOut));
    }
  }

  MLP.prototype._forwardLayer = function (a, l, cache) {
    const inSize = this.sizes[l];
    const outSize = this.sizes[l + 1];
    const Wl = this.W[l];
    const bl = this.b[l];
    const N = a.length;
    const isOutput = l === this.L - 1;
    const f = isOutput ? ACTIVATIONS.sigmoid.f : this.act.f;
    const out = new Array(N);
    const zs = cache ? new Array(N) : null;

    for (let i = 0; i < N; i++) {
      const ai = a[i];
      const zi = new Float64Array(outSize);
      const oi = new Float64Array(outSize);
      for (let j = 0; j < outSize; j++) {
        const wj = Wl[j];
        let s = bl[j];
        for (let k = 0; k < inSize; k++) s += wj[k] * ai[k];
        zi[j] = s;
        oi[j] = f(s);
      }
      out[i] = oi;
      if (zs) zs[i] = zi;
    }
    return { out: out, z: zs };
  };

  /** 前向传播，返回每一层的预激活 z 与激活 a，用于反向传播 */
  MLP.prototype.forward = function (X) {
    let a = X;
    const zs = [];
    const as = [];
    const aIns = [X];
    for (let l = 0; l < this.L; l++) {
      const r = this._forwardLayer(a, l, true);
      zs.push(r.z);
      as.push(r.out);
      aIns.push(r.out);
      a = r.out;
    }
    return { zs: zs, as: as, aIns: aIns };
  };

  /** 只做预测，不保留中间缓存（用于决策边界的高频网格预测） */
  MLP.prototype.predict = function (X) {
    let a = X;
    for (let l = 0; l < this.L; l++) {
      a = this._forwardLayer(a, l, false).out;
    }
    return a;
  };

  /** 对扁平坐标 [x1,y1,x2,y2,...] 批量预测，返回每个点的概率（扁平数组） */
  MLP.prototype.predictFlat = function (coords, count) {
    const inSize = this.sizes[0];
    let a = new Float64Array(count * inSize);
    for (let i = 0; i < count; i++) {
      for (let k = 0; k < inSize; k++) a[i * inSize + k] = coords[i * inSize + k];
    }
    for (let l = 0; l < this.L; l++) {
      const fanIn = this.sizes[l];
      const fanOut = this.sizes[l + 1];
      const isOutput = l === this.L - 1;
      const f = isOutput ? ACTIVATIONS.sigmoid.f : this.act.f;
      const Wl = this.W[l];
      const bl = this.b[l];
      const out = new Float64Array(count * fanOut);
      for (let i = 0; i < count; i++) {
        const base = i * fanIn;
        const obase = i * fanOut;
        for (let j = 0; j < fanOut; j++) {
          const wj = Wl[j];
          let s = bl[j];
          for (let k = 0; k < fanIn; k++) s += wj[k] * a[base + k];
          out[obase + j] = f(s);
        }
      }
      a = out;
    }
    return a;
  };

  /** 计算二元交叉熵损失与全部梯度 */
  MLP.prototype.lossAndGrad = function (X, Y) {
    const N = X.length;
    const cache = this.forward(X);
    const p = cache.as[this.L - 1];
    let loss = 0;
    for (let i = 0; i < N; i++) {
      const pi = Math.min(1 - 1e-7, Math.max(1e-7, p[i][0]));
      loss += -(Y[i] * Math.log(pi) + (1 - Y[i]) * Math.log(1 - pi));
    }
    loss /= N;

    // 输出层：sigmoid + BCE 的导数就是 (p - y)
    let delta = new Array(N);
    for (let i = 0; i < N; i++) {
      const d = new Float64Array(1);
      d[0] = (p[i][0] - Y[i]) / N;
      delta[i] = d;
    }

    const gradW = new Array(this.L);
    const gradB = new Array(this.L);

    for (let l = this.L - 1; l >= 0; l--) {
      const fanIn = this.sizes[l];
      const fanOut = this.sizes[l + 1];
      const aPrev = cache.aIns[l];
      const dW = new Array(fanOut);
      const db = new Float64Array(fanOut);
      for (let j = 0; j < fanOut; j++) dW[j] = new Float64Array(fanIn);

      for (let i = 0; i < N; i++) {
        const di = delta[i];
        const ap = aPrev[i];
        for (let j = 0; j < fanOut; j++) {
          const d = di[j];
          if (d === 0) continue;
          db[j] += d;
          const row = dW[j];
          for (let k = 0; k < fanIn; k++) row[k] += d * ap[k];
        }
      }
      gradW[l] = dW;
      gradB[l] = db;

      if (l > 0) {
        const prevSize = this.sizes[l];
        const zPrev = cache.zs[l - 1];
        const aPrevLayer = cache.as[l - 1];
        const prevDelta = new Array(N);
        const act = this.act;
        for (let i = 0; i < N; i++) {
          const nd = new Float64Array(prevSize);
          const di = delta[i];
          for (let k = 0; k < prevSize; k++) {
            let s = 0;
            for (let j = 0; j < fanOut; j++) s += di[j] * this.W[l][j][k];
            nd[k] = s * act.df(zPrev[i][k], aPrevLayer[i][k]);
          }
          prevDelta[i] = nd;
        }
        delta = prevDelta;
      }
    }

    return { loss: loss, gradW: gradW, gradB: gradB };
  };

  MLP.prototype.evaluate = function (X, Y) {
    const p = this.predict(X);
    const N = X.length;
    let loss = 0;
    let correct = 0;
    for (let i = 0; i < N; i++) {
      const pi = Math.min(1 - 1e-7, Math.max(1e-7, p[i][0]));
      loss += -(Y[i] * Math.log(pi) + (1 - Y[i]) * Math.log(1 - pi));
      const pred = p[i][0] >= 0.5 ? 1 : 0;
      if (pred === Y[i]) correct++;
    }
    return { loss: loss / N, accuracy: correct / N };
  };

  MLP.prototype.paramCount = function () {
    let n = 0;
    for (let l = 0; l < this.L; l++) {
      n += this.sizes[l] * this.sizes[l + 1] + this.sizes[l + 1];
    }
    return n;
  };

  /* ---------------- 优化器 ---------------- */

  function zerosW(W) {
    return W.map(function (layer) {
      return layer.map(function (row) {
        return new Float64Array(row.length);
      });
    });
  }

  function zerosB(b) {
    return b.map(function (row) {
      return new Float64Array(row.length);
    });
  }

  function Optimizer(model, type) {
    this.type = type || 'adam';
    this.t = 0;
    this.mu = 0.9;
    this.b1 = 0.9;
    this.b2 = 0.999;
    this.eps = 1e-8;
    this.vW = null;
    this.vB = null;
    this.mW = null;
    this.mB = null;
    if (this.type === 'momentum') {
      this.vW = zerosW(model.W);
      this.vB = zerosB(model.b);
    } else if (this.type === 'adam') {
      this.mW = zerosW(model.W);
      this.mB = zerosB(model.b);
      this.vW = zerosW(model.W);
      this.vB = zerosB(model.b);
    }
  }

  /** 执行一步参数更新，返回本次更新的均方根（用于观察参数变化幅度） */
  Optimizer.prototype.step = function (model, grads, lr) {
    this.t++;
    let sq = 0;
    let count = 0;
    const self = this;
    const b1pow = 1 - Math.pow(this.b1, this.t);
    const b2pow = 1 - Math.pow(this.b2, this.t);

    function updateMatrix(param, grad, mState, vState) {
      for (let j = 0; j < param.length; j++) {
        const pj = param[j];
        const gj = grad[j];
        for (let k = 0; k < pj.length; k++) {
          const g = gj[k];
          let upd;
          if (self.type === 'sgd') {
            upd = -lr * g;
          } else if (self.type === 'momentum') {
            vState[j][k] = self.mu * vState[j][k] - lr * g;
            upd = vState[j][k];
          } else {
            const m = self.b1 * mState[j][k] + (1 - self.b1) * g;
            const v = self.b2 * vState[j][k] + (1 - self.b2) * g * g;
            mState[j][k] = m;
            vState[j][k] = v;
            upd = (-lr * (m / b1pow)) / (Math.sqrt(v / b2pow) + self.eps);
          }
          pj[k] += upd;
          sq += upd * upd;
          count++;
        }
      }
    }

    function updateVector(param, grad, mState, vState) {
      for (let j = 0; j < param.length; j++) {
        const g = grad[j];
        let upd;
        if (self.type === 'sgd') {
          upd = -lr * g;
        } else if (self.type === 'momentum') {
          vState[j] = self.mu * vState[j] - lr * g;
          upd = vState[j];
        } else {
          const m = self.b1 * mState[j] + (1 - self.b1) * g;
          const v = self.b2 * vState[j] + (1 - self.b2) * g * g;
          mState[j] = m;
          vState[j] = v;
          upd = (-lr * (m / b1pow)) / (Math.sqrt(v / b2pow) + self.eps);
        }
        param[j] += upd;
        sq += upd * upd;
        count++;
      }
    }

    for (let l = 0; l < model.L; l++) {
      if (this.type === 'sgd') {
        updateMatrix(model.W[l], grads.gradW[l], null, null);
        updateVector(model.b[l], grads.gradB[l], null, null);
      } else if (this.type === 'momentum') {
        updateMatrix(model.W[l], grads.gradW[l], null, this.vW[l]);
        updateVector(model.b[l], grads.gradB[l], null, this.vB[l]);
      } else {
        updateMatrix(model.W[l], grads.gradW[l], this.mW[l], this.vW[l]);
        updateVector(model.b[l], grads.gradB[l], this.mB[l], this.vB[l]);
      }
    }

    return count ? Math.sqrt(sq / count) : 0;
  };

  /* ---------------- 数据生成 ---------------- */

  function generateData(type, n, noise, seed) {
    const rand = mulberry32(seed == null ? 7 : seed);
    const half = Math.floor(n / 2);
    const X = [];
    const Y = [];
    const scaleNoise = noise == null ? 0.12 : noise;

    for (let i = 0; i < half * 2; i++) {
      const cls = i % 2;
      let x;
      let y;

      if (type === 'moons') {
        const t = rand() * Math.PI;
        if (cls === 0) {
          x = Math.cos(t);
          y = Math.sin(t);
        } else {
          x = 1 - Math.cos(t);
          y = 0.5 - Math.sin(t);
        }
        x = (x - 0.5) / 1.5;
        y = (y - 0.25) / 1.5;
        x += gaussian(rand) * scaleNoise;
        y += gaussian(rand) * scaleNoise;
      } else if (type === 'circles') {
        const r = (cls === 0 ? 0.38 : 0.88) + gaussian(rand) * scaleNoise * 0.6;
        const th = rand() * Math.PI * 2;
        x = r * Math.cos(th);
        y = r * Math.sin(th);
      } else if (type === 'spiral') {
        const r = 0.12 + 0.88 * rand();
        const th = cls * Math.PI + r * 3.4 * Math.PI + gaussian(rand) * scaleNoise;
        x = r * Math.cos(th);
        y = r * Math.sin(th);
      } else {
        // xor
        x = rand() * 2 - 1 + gaussian(rand) * scaleNoise * 0.4;
        y = rand() * 2 - 1 + gaussian(rand) * scaleNoise * 0.4;
        const label = x * y >= 0 ? 1 : 0;
        X.push(Float64Array.from([x, y]));
        Y.push(label);
        continue;
      }

      X.push(Float64Array.from([x, y]));
      Y.push(cls);
    }

    // 打乱顺序
    const order = shuffleIndices(X.length, rand);
    const Xs = new Array(X.length);
    const Ys = new Float64Array(X.length);
    for (let i = 0; i < order.length; i++) {
      Xs[i] = X[order[i]];
      Ys[i] = Y[order[i]];
    }
    return { X: Xs, Y: Ys, name: type };
  }

  function shuffleIndices(n, rand) {
    const idx = new Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = idx[i];
      idx[i] = idx[j];
      idx[j] = tmp;
    }
    return idx;
  }

  function subset(X, Y, idx) {
    const Xb = new Array(idx.length);
    const Yb = new Float64Array(idx.length);
    for (let i = 0; i < idx.length; i++) {
      Xb[i] = X[idx[i]];
      Yb[i] = Y[idx[i]];
    }
    return { X: Xb, Y: Yb };
  }

  /* ---------------- 一次性训练（用于学习率对比） ---------------- */

  function trainFull(options) {
    const data = options.data;
    const model = new MLP(options.sizes, {
      activation: options.activation,
      seed: options.seed == null ? 42 : options.seed,
    });
    const optimizer = new Optimizer(model, options.optimizer);
    const rand = mulberry32((options.seed == null ? 42 : options.seed) + 991);
    const batchSize = Math.min(options.batchSize || 16, data.X.length);
    const history = { loss: [], acc: [] };
    const epochs = options.epochs || 200;

    for (let e = 0; e < epochs; e++) {
      const order = shuffleIndices(data.X.length, rand);
      for (let s = 0; s < order.length; s += batchSize) {
        const idx = order.slice(s, s + batchSize);
        const batch = subset(data.X, data.Y, idx);
        const grads = model.lossAndGrad(batch.X, batch.Y);
        optimizer.step(model, grads, options.lr);
      }
      const ev = model.evaluate(data.X, data.Y);
      history.loss.push(ev.loss);
      history.acc.push(ev.accuracy);
      if (options.onEpoch) options.onEpoch(e, ev);
    }
    return { model: model, history: history };
  }

  function computeStats(model) {
    let total = 0;
    let wSq = 0;
    let bSq = 0;
    let maxAbs = 0;
    const perLayer = [];
    for (let l = 0; l < model.L; l++) {
      let lSq = 0;
      let lCount = 0;
      for (let j = 0; j < model.W[l].length; j++) {
        const row = model.W[l][j];
        for (let k = 0; k < row.length; k++) {
          const w = row[k];
          wSq += w * w;
          lSq += w * w;
          if (Math.abs(w) > maxAbs) maxAbs = Math.abs(w);
          lCount++;
        }
      }
      for (let j = 0; j < model.b[l].length; j++) {
        const b = model.b[l][j];
        bSq += b * b;
        if (Math.abs(b) > maxAbs) maxAbs = Math.abs(b);
        lCount++;
      }
      total += lCount;
      perLayer.push({ norm: Math.sqrt(lSq), count: lCount });
    }
    return {
      total: total,
      weightNorm: Math.sqrt(wSq),
      biasNorm: Math.sqrt(bSq),
      maxAbs: maxAbs,
      perLayer: perLayer,
    };
  }

  return {
    MLP: MLP,
    Optimizer: Optimizer,
    ACTIVATIONS: ACTIVATIONS,
    generateData: generateData,
    shuffleIndices: shuffleIndices,
    subset: subset,
    trainFull: trainFull,
    computeStats: computeStats,
    mulberry32: mulberry32,
    gaussian: gaussian,
  };
});
