import React, { useMemo, useState } from "react";

// ==========================
// Utilities
// ==========================
// Seeded PRNG (Mulberry32)
function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashString(s) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function softmaxRowWise(mat) {
  const out = mat.map((row) => {
    const m = Math.max(...row);
    const exps = row.map((v) => Math.exp(v - m));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / (sum || 1));
  });
  return out;
}

function matmul(A, B) {
  const n = A.length;
  const m = A[0].length;
  const p = B[0].length;
  const out = Array.from({ length: n }, () => Array(p).fill(0));
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < m; k++) {
      const aik = A[i][k];
      for (let j = 0; j < p; j++) out[i][j] += aik * B[k][j];
    }
  }
  return out;
}

function transpose(A) {
  const n = A.length,
    m = A[0].length;
  const T = Array.from({ length: m }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) T[j][i] = A[i][j];
  return T;
}

function add(A, B) {
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}

// NOTE: renamed from `scale` to avoid name collision with local variables
function scaleMatrix(A, s) {
  return A.map((row) => row.map((v) => v * s));
}

function makeMatrix(rng, rows, cols, scaleStd = 0.2) {
  const out = Array.from({ length: rows }, () => Array(cols).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      // Xavier-like init
      const val = (rng() * 2 - 1) * scaleStd;
      out[i][j] = val;
    }
  }
  return out;
}

function chunkHeads(X, nHeads) {
  // X: [T, d] -> [H, T, dHead]
  const T = X.length,
    d = X[0].length;
  const dHead = Math.floor(d / nHeads);
  const heads = [];
  for (let h = 0; h < nHeads; h++) {
    const start = h * dHead;
    const end = start + dHead;
    const slice = X.map((row) => row.slice(start, end));
    heads.push(slice);
  }
  return heads;
}

function combineHeads(heads) {
  // heads: [H, T, dHead] -> [T, H*dHead]
  const H = heads.length,
    T = heads[0].length,
    dHead = heads[0][0].length;
  const out = Array.from({ length: T }, () => Array(H * dHead).fill(0));
  for (let t = 0; t < T; t++) {
    for (let h = 0; h < H; h++) {
      for (let j = 0; j < dHead; j++) out[t][h * dHead + j] = heads[h][t][j];
    }
  }
  return out;
}

function HeatCell({ value }) {
  // value in [0,1]
  const bg = `hsl(${(1 - value) * 240}, 70%, ${30 + value * 40}%)`;
  return <div className="w-6 h-6 md:w-7 md:h-7" style={{ backgroundColor: bg }} />;
}

function AttentionHeatmap({ tokens, weights }) {
  // weights: [T, T] row-normalized
  const T = tokens.length;
  return (
    <div className="overflow-auto">
      <div className="text-sm text-muted-foreground mb-2">
        Attention heatmap (rows = Query t, cols = Key t')
      </div>
      <div className="grid" style={{ gridTemplateColumns: `repeat(${T + 1}, minmax(1rem, auto))` }}>
        <div />
        {tokens.map((tk, j) => (
          <div key={"col-" + j} className="text-xs text-center px-1 truncate">
            {tk}
          </div>
        ))}
        {weights.map((row, i) => (
          <React.Fragment key={"r-" + i}>
            <div className="text-xs text-right pr-1 truncate">{tokens[i]}</div>
            {row.map((v, j) => (
              <HeatCell key={`cell-${i}-${j}`} value={Math.max(0, Math.min(1, v))} />
            ))}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

// ==========================
// Tiny Test Harness (runtime tests)
// ==========================
function approx(a, b, eps = 1e-6) {
  return Math.abs(a - b) <= eps;
}
function approxArray(A, B, eps = 1e-6) {
  if (A.length !== B.length) return false;
  for (let i = 0; i < A.length; i++) if (!approx(A[i], B[i], eps)) return false;
  return true;
}
function approxMatrix(A, B, eps = 1e-6) {
  if (A.length !== B.length) return false;
  for (let i = 0; i < A.length; i++) if (!approxArray(A[i], B[i], eps)) return false;
  return true;
}

function runSelfTests() {
  const tests = [];

  // 1) scaleMatrix basic
  (function () {
    const A = [
      [1, 2],
      [3, 4],
    ];
    const out = scaleMatrix(A, 0.5);
    const exp = [
      [0.5, 1],
      [1.5, 2],
    ];
    tests.push({ name: "scaleMatrix halves entries", pass: approxMatrix(out, exp) });
  })();

  // 2) softmax row-wise sanity
  (function () {
    const out = softmaxRowWise([
      [0, 0],
      [1, 2],
    ]);
    const pass = approxArray(out[0], [0.5, 0.5]) && out[1][1] > out[1][0];
    tests.push({ name: "softmaxRowWise stable & increasing", pass });
  })();

  // 3) matmul identity
  (function () {
    const I = [
      [1, 0],
      [0, 1],
    ];
    const X = [
      [2, 3],
      [4, 5],
    ];
    const out = matmul(X, I);
    tests.push({ name: "matmul(X, I) == X", pass: approxMatrix(out, X) });
  })();

  // 4) causal masking effect on softmax
  (function () {
    const Q = [
      [1, 0],
      [0, 1],
      [1, 1],
    ];
    const K = [
      [1, 0],
      [0, 1],
      [1, 1],
    ];
    const KT = transpose(K);
    let scores = matmul(Q, KT); // [3,3]
    scores = scaleMatrix(scores, 1 / Math.sqrt(2));

    // Apply causal mask
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) if (j > i) scores[i][j] = -1e9;
    const W = softmaxRowWise(scores);
    const pass = W[0][1] < 1e-6 && W[0][2] < 1e-6 && W[1][2] < 1e-6; // masked positions ~ 0
    tests.push({ name: "causal mask zeros-out future keys", pass });
  })();

  return tests;
}

// ==========================
// Main Component
// ==========================
export default function TransformerAttentionPlayground() {
  const [text, setText] = useState("xin chao transformer! day la demo.");
  const [tokenMode, setTokenMode] = useState("char"); // 'char' | 'word'
  const [dModel, setDModel] = useState(32);
  const [nHeads, setNHeads] = useState(2);
  const [causal, setCausal] = useState(true);
  const [seed, setSeed] = useState(42);
  const [tests, setTests] = useState(null);

  const tokens = useMemo(() => {
    const arr = tokenMode === "word" ? text.split(/\s+/).filter(Boolean) : [...text];
    return arr.slice(0, 40); // keep it small
  }, [text, tokenMode]);

  const { attnAvg, attnPerHead } = useMemo(() => {
    const T = tokens.length;
    const d = Math.max(4, Math.floor(dModel / 4) * 4); // divisible by 4
    const H = Math.max(1, Math.min(nHeads, d / 4));
    const dHead = Math.floor(d / H);

    // Seed RNG based on seed + tokens (deterministic per seed)
    const rng = mulberry32(hashString(tokens.join("|") + "#" + seed));

    // Embeddings: simple deterministic per token
    const embed = tokens.map((tk) => {
      const r = mulberry32(hashString("emb:" + tk + ":" + seed));
      return Array.from({ length: d }, () => (r() * 2 - 1) * 0.5);
    });

    // Projection matrices
    const Wq = makeMatrix(rng, d, d);
    const Wk = makeMatrix(rng, d, d);
    const Wv = makeMatrix(rng, d, d);

    // X: [T,d]
    const X = embed;
    const Qall = matmul(X, Wq); // [T,d]
    const Kall = matmul(X, Wk);
    const Vall = matmul(X, Wv);

    // Split heads
    const Qh = chunkHeads(Qall, H);
    const Kh = chunkHeads(Kall, H);
    const Vh = chunkHeads(Vall, H);

    const perHeadWeights = [];

    for (let h = 0; h < H; h++) {
      const Q = Qh[h];
      const K = Kh[h];
      const KT = transpose(K);

      // scores = Q @ K^T / sqrt(dHead)
      let scores = matmul(Q, KT);
      scores = scaleMatrix(scores, 1 / Math.sqrt(dHead));

      // Apply mask
      if (causal) {
        for (let i = 0; i < T; i++) {
          for (let j = 0; j < T; j++) if (j > i) scores[i][j] = -1e9;
        }
      }

      const weights = softmaxRowWise(scores);
      perHeadWeights.push(weights);
    }

    // Average over heads for display
    const attnAvg = Array.from({ length: tokens.length }, () => Array(tokens.length).fill(0));
    for (let h = 0; h < perHeadWeights.length; h++) {
      for (let i = 0; i < tokens.length; i++) {
        for (let j = 0; j < tokens.length; j++) {
          attnAvg[i][j] += perHeadWeights[h][i][j] / perHeadWeights.length;
        }
      }
    }

    return { attnAvg, attnPerHead: perHeadWeights };
  }, [tokens, dModel, nHeads, causal, seed]);

  const headTabs = attnPerHead.map((w, i) => (
    <div key={"h-" + i} className="mb-6">
      <div className="font-medium mb-2">Head {i + 1}</div>
      <AttentionHeatmap tokens={tokens} weights={w} />
    </div>
  ));

  return (
    <div className="p-4 md:p-6 max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl md:text-3xl font-semibold">Transformer Attention Playground</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Nhập chuỗi ký tự hoặc từ. Chọn số head, kích thước mô hình và kiểu mask. App tính toán self-attention (scaled dot-product) và hiển thị heatmap.
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-4 mb-6">
        <div className="md:col-span-2 bg-white/60 dark:bg-zinc-900/40 rounded-2xl p-4 shadow">
          <label className="text-sm font-medium">Văn bản đầu vào</label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="w-full mt-2 rounded-xl border p-3 min-h-[96px]"
            placeholder="Nhập văn bản..."
          />

          <div className="flex flex-wrap gap-3 mt-4 items-center">
            <div className="flex items-center gap-2">
              <input type="radio" id="char" checked={tokenMode === "char"} onChange={() => setTokenMode("char")} />
              <label htmlFor="char" className="text-sm">Token hoá theo ký tự</label>
            </div>
            <div className="flex items-center gap-2">
              <input type="radio" id="word" checked={tokenMode === "word"} onChange={() => setTokenMode("word")} />
              <label htmlFor="word" className="text-sm">Token hoá theo từ</label>
            </div>
            <div className="flex items-center gap-2">
              <input type="checkbox" id="causal" checked={causal} onChange={(e) => setCausal(e.target.checked)} />
              <label htmlFor="causal" className="text-sm">Causal mask (không nhìn tương lai)</label>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 mt-4">
            <div>
              <label className="text-sm">d_model: {dModel}</label>
              <input type="range" min={16} max={128} step={4} value={dModel} onChange={(e) => setDModel(parseInt(e.target.value))} className="w-full" />
            </div>
            <div>
              <label className="text-sm">n_heads: {nHeads}</label>
              <input type="range" min={1} max={8} step={1} value={nHeads} onChange={(e) => setNHeads(parseInt(e.target.value))} className="w-full" />
            </div>
          </div>

          <div className="mt-4 flex gap-3">
            <button className="px-3 py-2 rounded-xl bg-black text-white" onClick={() => setSeed((s) => s + 1)}>Re-seed</button>
            <div className="text-sm text-muted-foreground self-center">Seed: {seed}</div>
          </div>
        </div>

        <div className="bg-white/60 dark:bg-zinc-900/40 rounded-2xl p-4 shadow">
          <h3 className="font-medium mb-2">Tóm tắt</h3>
          <ul className="list-disc pl-4 text-sm space-y-1">
            <li>
              <b>Q, K, V</b> được chiếu tuyến tính từ embedding token.
            </li>
            <li>
              <b>Attention</b> = softmax((QK^T)/√d_head + mask).
            </li>
            <li>
              <b>Causal</b>: tam giác dưới; mỗi vị trí chỉ nhìn quá khứ.
            </li>
            <li>
              Nhiều <b>head</b>: học phụ thuộc khác nhau; đồ thị hiển thị theo từng head và trung bình.
            </li>
          </ul>
        </div>
      </div>

      <div className="mb-8 bg-white/60 dark:bg-zinc-900/40 rounded-2xl p-4 shadow">
        <h3 className="font-medium mb-4">Heatmap (trung bình các head)</h3>
        {tokens.length > 0 ? (
          <AttentionHeatmap tokens={tokens} weights={attnAvg} />
        ) : (
          <div className="text-sm text-muted-foreground">Nhập văn bản để xem attention.</div>
        )}
      </div>

      <div className="mb-8 bg-white/60 dark:bg-zinc-900/40 rounded-2xl p-4 shadow">
        <h3 className="font-medium mb-4">Từng Head</h3>
        {attnPerHead.length > 1 ? headTabs : <div className="text-sm text-muted-foreground">Chỉ có 1 head.</div>}
      </div>

      {/* ===== Dev/Test Panel ===== */}
      <div className="mb-8 bg-white/60 dark:bg-zinc-900/40 rounded-2xl p-4 shadow">
        <h3 className="font-medium mb-3">Dev / Tests</h3>
        <button
          className="px-3 py-2 rounded-xl bg-emerald-600 text-white"
          onClick={() => setTests(runSelfTests())}
        >
          Run tests
        </button>
        {tests && (
          <ul className="mt-3 text-sm space-y-1">
            {tests.map((t, i) => (
              <li key={i} className="flex items-center gap-2">
                <span className={`inline-block w-2 h-2 rounded-full ${t.pass ? "bg-emerald-500" : "bg-rose-500"}`} />
                <span>{t.name}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="opacity-70 text-xs">
        *Đây là mô hình <b>đồ chơi</b> không huấn luyện. Mục tiêu là minh hoạ self-attention & mask. Bạn có thể thay đổi token hoá, số head, kích thước mô hình và reseed để thấy heatmap khác nhau.
      </div>
    </div>
  );
}
