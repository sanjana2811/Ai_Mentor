import { afterEach, beforeEach, test } from "node:test";
import assert from "node:assert/strict";

process.env.NODE_ENV = "test";
process.env.GEMINI_API_KEY = "fake-gemini-key";
process.env.GROQ_API_KEY = "fake-groq-key";
process.env.GEMINI_CHAT_MODEL = "gemini-test-model";
process.env.GROQ_CHAT_MODEL = "llama-test-model";

const {
  askGemini,
  classifyGeminiError,
  deps,
  resetGeminiClientForTesting,
  setGeminiClientForTesting,
} = await import("../services/geminiChatService.js");

const original = {
  fetch: deps.fetch,
  timeoutMs: deps.timeoutMs,
  logger: deps.logger,
  geminiModel: process.env.GEMINI_CHAT_MODEL,
  groqModel: process.env.GROQ_CHAT_MODEL,
  groqKey: process.env.GROQ_API_KEY,
};
let logs;
let requests;

const geminiSuccess = (text = "Hello learner") => ({
  models: {
    generateContent: async () => ({ text, candidates: [] }),
  },
});
const groqSuccess = (text = "Groq fallback reply") => ({
  ok: true,
  status: 200,
  json: async () => ({ choices: [{ message: { content: text } }] }),
});
const groqFailure = (status = 503) => ({
  ok: false,
  status,
  json: async () => ({ error: { code: "PROVIDER_UNAVAILABLE" } }),
});

beforeEach(() => {
  logs = [];
  requests = [];
  deps.timeoutMs = 1000;
  deps.logger = {
    warn: (...args) => logs.push({ level: "warn", args }),
    info: (...args) => logs.push({ level: "info", args }),
  };
  deps.fetch = async (url, init) => {
    requests.push({ url: String(url), init });
    return groqSuccess();
  };
  process.env.GROQ_API_KEY = "fake-groq-key";
  process.env.GEMINI_CHAT_MODEL = "gemini-test-model";
  process.env.GROQ_CHAT_MODEL = "llama-test-model";
  setGeminiClientForTesting(geminiSuccess());
});

afterEach(() => {
  deps.fetch = original.fetch;
  deps.timeoutMs = original.timeoutMs;
  deps.logger = original.logger;
  process.env.GEMINI_CHAT_MODEL = original.geminiModel;
  process.env.GROQ_CHAT_MODEL = original.groqModel;
  process.env.GROQ_API_KEY = original.groqKey;
  resetGeminiClientForTesting();
});

test("Gemini success returns its reply without calling Groq", async () => {
  setGeminiClientForTesting(geminiSuccess("Gemini reply"));

  const result = await askGemini("learner context", "hello", { requestId: "req-1" });

  assert.deepEqual(result, {
    ok: true,
    text: "Gemini reply",
    moderated: false,
    provider: "gemini",
  });
  assert.equal(requests.length, 0);
});

test("Gemini uses the model configured by environment", async () => {
  let request;
  setGeminiClientForTesting({
    models: {
      generateContent: async (args) => {
        request = args;
        return { text: "Configured model reply", candidates: [] };
      },
    },
  });

  await askGemini("context", "hello");

  assert.equal(request.model, "gemini-test-model");
  assert.equal(request.config.httpOptions.timeout, deps.timeoutMs);
});

test("Gemini 500 falls back to Groq and returns Groq's reply", async () => {
  setGeminiClientForTesting({
    models: {
      generateContent: async () => {
        const error = new Error("private Gemini provider message");
        error.status = 500;
        throw error;
      },
    },
  });

  const result = await askGemini("learner context", "hello", { requestId: "req-2" });

  assert.deepEqual(result, {
    ok: true,
    text: "Groq fallback reply",
    moderated: false,
    provider: "groq",
  });
  assert.equal(requests.length, 1);
  assert.equal(requests[0].url, "https://api.groq.com/openai/v1/chat/completions");
  assert.equal(requests[0].init.headers.Authorization, "Bearer fake-groq-key");
  const body = JSON.parse(requests[0].init.body);
  assert.equal(body.model, "llama-test-model");
  assert.equal(body.messages[0].role, "system");
  assert.equal(body.messages.at(-1).content, "hello");
  assert.ok(logs.some((entry) => entry.level === "warn"));
  assert.ok(!JSON.stringify(logs).includes("private Gemini provider message"));
});

test("Groq fallback preserves prior conversation turns", async () => {
  setGeminiClientForTesting({
    models: { generateContent: async () => { throw Object.assign(new Error("down"), { status: 500 }); } },
  });

  await askGemini("context", "current question", {
    history: [
      { role: "user", text: "previous question" },
      { role: "model", text: "previous answer" },
    ],
  });

  const body = JSON.parse(requests[0].init.body);
  assert.deepEqual(body.messages.slice(-3), [
    { role: "user", content: "previous question" },
    { role: "assistant", content: "previous answer" },
    { role: "user", content: "current question" },
  ]);
});

test("both providers failing returns a safe unavailable result", async () => {
  setGeminiClientForTesting({
    models: { generateContent: async () => { throw Object.assign(new Error("gemini secret"), { status: 500 }); } },
  });
  deps.fetch = async (url, init) => {
    requests.push({ url: String(url), init });
    return groqFailure(503);
  };

  const result = await askGemini("context", "sensitive prompt", { requestId: "req-3" });

  assert.equal(result.ok, false);
  assert.equal(result.category, "unavailable");
  assert.match(result.message, /temporarily unavailable/i);
  assert.equal(logs.filter((entry) => entry.level === "warn").length, 2);
  assert.ok(!JSON.stringify(logs).includes("gemini secret"));
  assert.ok(!JSON.stringify(logs).includes("sensitive prompt"));
});

test("a Gemini timeout still tries Groq", async () => {
  setGeminiClientForTesting({
    models: {
      generateContent: async () => {
        const error = new Error("timed out");
        error.name = "TimeoutError";
        throw error;
      },
    },
  });

  const result = await askGemini("context", "hello");

  assert.equal(result.ok, true);
  assert.equal(result.provider, "groq");
});

test("a Groq timeout is classified and returned safely", async () => {
  setGeminiClientForTesting({
    models: { generateContent: async () => { throw Object.assign(new Error("down"), { status: 500 }); } },
  });
  deps.timeoutMs = 15;
  deps.fetch = (_url, init) =>
    new Promise((_resolve, reject) => {
      init.signal.addEventListener("abort", () => {
        reject(new DOMException("aborted", "AbortError"));
      }, { once: true });
    });

  const result = await askGemini("context", "hello");

  assert.equal(result.ok, false);
  assert.equal(result.category, "timeout");
});

test("classifyGeminiError keeps existing status classification", () => {
  assert.equal(classifyGeminiError({ status: 429 }).category, "rate_limited");
  assert.equal(classifyGeminiError({ status: 500 }).category, "unavailable");
  assert.equal(classifyGeminiError({ name: "TimeoutError" }).category, "timeout");
});
