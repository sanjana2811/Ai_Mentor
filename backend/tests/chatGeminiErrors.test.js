import { after, afterEach, beforeEach, test } from "node:test";
import assert from "node:assert/strict";

process.env.NODE_ENV = "test";
process.env.GEMINI_API_KEY = "fake-test-key";
delete process.env.GROQ_API_KEY;

const { default: User } = await import("../models/User.js");
const { default: Preference } = await import("../models/Preference.js");
const { chatWithAssistant } = await import("../controllers/chatController.js");
const { deps } = await import("../services/geminiChatService.js");

const SECRET = "SECRET-PROMPT-TEXT-123";
const originals = {
  userFindByPk: User.findByPk,
  preferenceFindOne: Preference.findOne,
  fetch: globalThis.fetch,
  logger: deps.logger,
};
let logs;

const response = () => {
  const state = { statusCode: 200, body: undefined };
  return {
    state,
    res: {
      status(code) {
        state.statusCode = code;
        return this;
      },
      json(body) {
        state.body = body;
        return this;
      },
    },
  };
};

const stubGemini = (status, body) => {
  globalThis.fetch = async () =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    });
};

const chat = async () => {
  const { res, state } = response();
  await chatWithAssistant({ id: "req-42", user: { id: 1 }, body: { message: SECRET } }, res);
  return state;
};

beforeEach(() => {
  logs = [];
  deps.logger = { warn: (...args) => logs.push(args), error: (...args) => logs.push(args) };
  User.findByPk = async () => ({ id: 1, name: "Sam", purchasedCourses: [], settings: {} });
  Preference.findOne = async () => null;
});

afterEach(() => {
  globalThis.fetch = originals.fetch;
  deps.logger = originals.logger;
});

after(() => {
  User.findByPk = originals.userFindByPk;
  Preference.findOne = originals.preferenceFindOne;
});

test("chat returns 200 with the reply and extracts the route on Gemini success", async () => {
  stubGemini(200, {
    candidates: [
      { content: { role: "model", parts: [{ text: "Open settings.\nROUTE:/settings" }] } },
    ],
  });

  const { statusCode, body } = await chat();

  assert.equal(statusCode, 200);
  assert.deepEqual(body, { reply: "Open settings.", route: "/settings" });
  assert.equal(logs.length, 0);
});

const failures = [
  ["429", 429, "RESOURCE_EXHAUSTED", "rate_limited", /too many requests/],
  ["500", 500, "INTERNAL", "unavailable", /temporarily unavailable/],
  ["503", 503, "UNAVAILABLE", "unavailable", /temporarily unavailable/],
  ["authentication failure", 401, "UNAUTHENTICATED", "auth", /currently unavailable/],
];

for (const [label, status, id, category] of failures) {
  test(`chat returns structured HTTP 503 when both providers fail on ${label}`, async () => {
    stubGemini(status, { error: { code: status, status: id, message: SECRET } });

    const { statusCode, body } = await chat();

    assert.equal(statusCode, 503);
    assert.deepEqual(body, {
      success: false,
      code: "AI_PROVIDER_UNAVAILABLE",
      message: "The AI service is temporarily unavailable. Please try again.",
      requestId: "req-42",
    });
    assert.ok(!JSON.stringify(body).includes(SECRET));
    assert.equal(logs.length, 1);
    assert.equal(logs[0][0].category, category);
    assert.equal(logs[0][0].status, status);
    assert.equal(logs[0][0].requestId, "req-42");
    assert.equal(logs[0][0].provider, "gemini");
    assert.ok(!JSON.stringify(logs).includes(SECRET));
  });
}

test("chat returns structured HTTP 503 on Gemini network failure when Groq is not configured", async () => {
  globalThis.fetch = async () => {
    throw new TypeError("fetch failed");
  };

  const { statusCode, body } = await chat();

  assert.equal(statusCode, 503);
  assert.equal(body.code, "AI_PROVIDER_UNAVAILABLE");
  assert.equal(body.requestId, "req-42");
  assert.equal(logs[0][0].category, "network");
});
