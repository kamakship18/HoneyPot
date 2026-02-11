"use client";

import React, { useMemo, useState, useRef, useEffect } from "react";

type Message = {
  role: "scammer" | "agent";
  content: string;
  timestamp: string;
};

type ExtractedIntel = {
  upi_ids: string[];
  links: string[];
  bank_accounts: string[];
  phone_numbers: string[];
  tactics: string[];
};

type SessionData = {
  session_id: string;
  turn_count: number;
  scam_detected: boolean;
  extracted_intelligence: ExtractedIntel;
  conversation_history: Message[];
  intel_count: number;
};

export default function Page() {
  const [sessionId, setSessionId] = useState(`scammer-test-${Date.now()}`);
  const [messageText, setMessageText] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [extractedIntel, setExtractedIntel] = useState<ExtractedIntel>({
    upi_ids: [],
    links: [],
    bank_accounts: [],
    phone_numbers: [],
    tactics: [],
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [conversationEnded, setConversationEnded] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const apiBase = useMemo(() => {
    return process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  async function sendMessage() {
    if (!messageText.trim() || loading) return;

    const userMessage: Message = {
      role: "scammer",
      content: messageText.trim(),
      timestamp: new Date().toISOString(),
    };

    // Add user message immediately
    setMessages((prev) => [...prev, userMessage]);
    setMessageText("");
    setLoading(true);
    setError(null);

    try {
      const headers: Record<string, string> = { "Content-Type": "application/json" };

      const r = await fetch(`${apiBase}/v1/chat`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          sessionId: sessionId,
          message: {
            sender: "scammer",
            text: userMessage.content,
            timestamp: userMessage.timestamp,
          },
          conversationHistory: messages.map((m) => ({
            role: m.role === "scammer" ? "scammer" : "agent",
            content: m.content,
          })),
          metadata: {},
        }),
      });

      if (!r.ok) {
        const text = await r.text();
        let msg = text || `HTTP ${r.status}`;
        try {
          const j = JSON.parse(text) as { detail?: string };
          if (j?.detail) msg = j.detail;
        } catch {
          /* */
        }
        throw new Error(msg);
      }

      const response = (await r.json()) as { status: string; reply: string };

      // Add agent response
      const agentMessage: Message = {
        role: "agent",
        content: response.reply,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, agentMessage]);

      // Fetch updated intelligence
      await fetchIntelligence();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  async function fetchIntelligence() {
    try {
      const r = await fetch(`${apiBase}/debug/session/${sessionId}`);
      if (r.ok) {
        const data = (await r.json()) as SessionData;
        setExtractedIntel(data.extracted_intelligence || {
          upi_ids: [],
          links: [],
          bank_accounts: [],
          phone_numbers: [],
          tactics: [],
        });

        // Check if conversation should end (very lenient - allow continuous conversations)
        // Only end if we've extracted a LOT of intel (20+ items) or had MANY turns (100+)
        // This allows for extended testing and conversation
        if (data.intel_count >= 20 || data.turn_count >= 100) {
          setConversationEnded(true);
        } else {
          // Reset ended state if thresholds not met (in case it was set earlier)
          setConversationEnded(false);
        }
      }
    } catch (e) {
      // Silently fail - intelligence fetch is optional
    }
  }

  function startNewSession() {
    setSessionId(`scammer-test-${Date.now()}`);
    setMessages([]);
    setExtractedIntel({
      upi_ids: [],
      links: [],
      bank_accounts: [],
      phone_numbers: [],
      tactics: [],
    });
    setError(null);
    setConversationEnded(false);
    inputRef.current?.focus();
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  const intelCount = useMemo(() => {
    return (
      extractedIntel.upi_ids.length +
      extractedIntel.links.length +
      extractedIntel.bank_accounts.length +
      extractedIntel.phone_numbers.length +
      extractedIntel.tactics.length
    );
  }, [extractedIntel]);

  return (
    <main
      style={{
        fontFamily: "ui-sans-serif, system-ui, -apple-system",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "#f5f5f5",
      }}
    >
      {/* Header */}
      <header
        style={{
          background: "#fff",
          borderBottom: "1px solid #e0e0e0",
          padding: "16px 24px",
          boxShadow: "0 2px 4px rgba(0,0,0,0.05)",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h1 style={{ margin: 0, fontSize: "20px", fontWeight: 600 }}>🎭 Scammer Test Interface</h1>
            <p style={{ margin: "4px 0 0", fontSize: "14px", color: "#666" }}>
              Act as a scammer and chat with the AI agent. Watch it extract intelligence.
            </p>
          </div>
          <button
            onClick={startNewSession}
            style={{
              padding: "8px 16px",
              borderRadius: "8px",
              border: "1px solid #ddd",
              background: "#fff",
              cursor: "pointer",
              fontSize: "14px",
            }}
          >
            🆕 New Session
          </button>
        </div>
      </header>

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* Chat Area */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", background: "#fff" }}>
          {/* Messages */}
          <div
            style={{
              flex: 1,
              overflowY: "auto",
              padding: "24px",
              display: "flex",
              flexDirection: "column",
              gap: "16px",
            }}
          >
            {messages.length === 0 && (
              <div
                style={{
                  textAlign: "center",
                  color: "#999",
                  marginTop: "40px",
                  padding: "20px",
                }}
              >
                <p style={{ fontSize: "16px", marginBottom: "8px" }}>💬 Start the conversation</p>
                <p style={{ fontSize: "14px" }}>Type a message as a scammer and see how the AI agent responds</p>
                <div style={{ marginTop: "24px", fontSize: "13px", color: "#bbb" }}>
                  <p>Example messages:</p>
                  <ul style={{ listStyle: "none", padding: 0, margin: "8px 0" }}>
                    <li>• "KYC pending. Your account will be blocked. Pay ₹99 to sbihelpdesk@okicici"</li>
                    <li>• "Congrats! You won 12 lakh lottery. Pay processing fee ₹4,999 via UPI: luckyprize@paytm"</li>
                    <li>• "Parcel stuck customs. Pay ₹2,150 duty. Send to UPI: customsfee@oksbi"</li>
                  </ul>
                </div>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div
                key={idx}
                style={{
                  display: "flex",
                  justifyContent: msg.role === "scammer" ? "flex-end" : "flex-start",
                }}
              >
                <div
                  style={{
                    maxWidth: "70%",
                    padding: "12px 16px",
                    borderRadius: "16px",
                    background: msg.role === "scammer" ? "#dc2626" : "#f3f4f6",
                    color: msg.role === "scammer" ? "#fff" : "#111",
                    border: msg.role === "scammer" ? "none" : "1px solid #e5e7eb",
                  }}
                >
                  <div style={{ fontSize: "12px", opacity: 0.7, marginBottom: "4px" }}>
                    {msg.role === "scammer" ? "🎭 You (Scammer)" : "🤖 AI Agent (Aman)"}
                  </div>
                  <div style={{ whiteSpace: "pre-wrap", lineHeight: "1.5" }}>{msg.content}</div>
                </div>
              </div>
            ))}

            {loading && (
              <div style={{ display: "flex", justifyContent: "flex-start" }}>
                <div
                  style={{
                    padding: "12px 16px",
                    borderRadius: "16px",
                    background: "#f3f4f6",
                    border: "1px solid #e5e7eb",
                    color: "#666",
                    fontSize: "14px",
                  }}
                >
                  🤖 AI Agent is typing...
                </div>
              </div>
            )}

            {error && (
              <div
                style={{
                  padding: "12px 16px",
                  borderRadius: "8px",
                  background: "#fee2e2",
                  border: "1px solid #fecaca",
                  color: "#991b1b",
                }}
              >
                ❌ Error: {error}
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div
            style={{
              borderTop: "1px solid #e0e0e0",
              padding: "16px 24px",
              background: "#fff",
            }}
          >
            {conversationEnded && (
              <div
                style={{
                  padding: "12px",
                  marginBottom: "12px",
                  borderRadius: "8px",
                  background: "#fef3c7",
                  border: "1px solid #fde68a",
                  fontSize: "14px",
                }}
              >
                ⚠️ Conversation limit reached (20+ intelligence items or 100+ turns)
                <br />
                <button
                  onClick={() => setConversationEnded(false)}
                  style={{
                    marginTop: "8px",
                    padding: "4px 8px",
                    borderRadius: "4px",
                    border: "1px solid #f59e0b",
                    background: "#fff",
                    cursor: "pointer",
                    fontSize: "12px",
                  }}
                >
                  Continue Anyway
                </button>
              </div>
            )}
            <div style={{ display: "flex", gap: "8px", alignItems: "flex-end" }}>
              <textarea
                ref={inputRef}
                value={messageText}
                onChange={(e) => setMessageText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type your scam message here... (Press Enter to send, Shift+Enter for new line)"
                disabled={loading || conversationEnded}
                rows={2}
                style={{
                  flex: 1,
                  padding: "12px",
                  borderRadius: "12px",
                  border: "1px solid #ddd",
                  fontSize: "14px",
                  resize: "none",
                  fontFamily: "inherit",
                  outline: "none",
                }}
              />
              <button
                onClick={sendMessage}
                disabled={loading || !messageText.trim() || conversationEnded}
                style={{
                  padding: "12px 24px",
                  borderRadius: "12px",
                  border: "none",
                  background: loading || !messageText.trim() || conversationEnded ? "#ccc" : "#dc2626",
                  color: "#fff",
                  cursor: loading || !messageText.trim() || conversationEnded ? "not-allowed" : "pointer",
                  fontSize: "14px",
                  fontWeight: 600,
                }}
              >
                {loading ? "Sending..." : "Send"}
              </button>
            </div>
          </div>
        </div>

        {/* Intelligence Panel */}
        <div
          style={{
            width: "350px",
            background: "#fff",
            borderLeft: "1px solid #e0e0e0",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              padding: "16px",
              borderBottom: "1px solid #e0e0e0",
              background: "#f9fafb",
            }}
          >
            <h2 style={{ margin: 0, fontSize: "16px", fontWeight: 600 }}>📊 Extracted Intelligence</h2>
            <p style={{ margin: "4px 0 0", fontSize: "12px", color: "#666" }}>
              Session: {sessionId.slice(-8)}
            </p>
            <div
              style={{
                marginTop: "8px",
                padding: "8px",
                borderRadius: "6px",
                background: intelCount > 0 ? "#dcfce7" : "#f3f4f6",
                textAlign: "center",
                fontSize: "14px",
                fontWeight: 600,
                color: intelCount > 0 ? "#166534" : "#666",
              }}
            >
              {intelCount} items extracted
            </div>
          </div>

          <div style={{ flex: 1, overflowY: "auto", padding: "16px" }}>
            {intelCount === 0 ? (
              <div style={{ textAlign: "center", color: "#999", padding: "20px", fontSize: "14px" }}>
                No intelligence extracted yet.
                <br />
                <br />
                The AI agent will extract:
                <br />• UPI IDs
                <br />• Phishing Links
                <br />• Bank Accounts
                <br />• Phone Numbers
                <br />• Suspicious Tactics
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                {extractedIntel.upi_ids.length > 0 && (
                  <div>
                    <div style={{ fontSize: "12px", fontWeight: 600, color: "#666", marginBottom: "8px" }}>
                      💳 UPI IDs
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                      {extractedIntel.upi_ids.map((upi, idx) => (
                        <div
                          key={idx}
                          style={{
                            padding: "8px",
                            background: "#f3f4f6",
                            borderRadius: "6px",
                            fontSize: "13px",
                            fontFamily: "monospace",
                          }}
                        >
                          {upi}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {extractedIntel.links.length > 0 && (
                  <div>
                    <div style={{ fontSize: "12px", fontWeight: 600, color: "#666", marginBottom: "8px" }}>
                      🔗 Phishing Links
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                      {extractedIntel.links.map((link, idx) => (
                        <div
                          key={idx}
                          style={{
                            padding: "8px",
                            background: "#f3f4f6",
                            borderRadius: "6px",
                            fontSize: "13px",
                            wordBreak: "break-all",
                            color: "#dc2626",
                          }}
                        >
                          {link}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {extractedIntel.bank_accounts.length > 0 && (
                  <div>
                    <div style={{ fontSize: "12px", fontWeight: 600, color: "#666", marginBottom: "8px" }}>
                      🏦 Bank Accounts
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                      {extractedIntel.bank_accounts.map((account, idx) => (
                        <div
                          key={idx}
                          style={{
                            padding: "8px",
                            background: "#f3f4f6",
                            borderRadius: "6px",
                            fontSize: "13px",
                            fontFamily: "monospace",
                          }}
                        >
                          {account}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {extractedIntel.phone_numbers.length > 0 && (
                  <div>
                    <div style={{ fontSize: "12px", fontWeight: 600, color: "#666", marginBottom: "8px" }}>
                      📞 Phone Numbers
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                      {extractedIntel.phone_numbers.map((phone, idx) => (
                        <div
                          key={idx}
                          style={{
                            padding: "8px",
                            background: "#f3f4f6",
                            borderRadius: "6px",
                            fontSize: "13px",
                            fontFamily: "monospace",
                          }}
                        >
                          {phone}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {extractedIntel.tactics.length > 0 && (
                  <div>
                    <div style={{ fontSize: "12px", fontWeight: 600, color: "#666", marginBottom: "8px" }}>
                      ⚠️ Suspicious Tactics
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                      {extractedIntel.tactics.map((tactic, idx) => (
                        <div
                          key={idx}
                          style={{
                            padding: "8px",
                            background: "#fef3c7",
                            borderRadius: "6px",
                            fontSize: "13px",
                          }}
                        >
                          {tactic}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Stats Footer */}
          <div
            style={{
              padding: "12px 16px",
              borderTop: "1px solid #e0e0e0",
              background: "#f9fafb",
              fontSize: "12px",
              color: "#666",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
              <span>Messages:</span>
              <span style={{ fontWeight: 600 }}>{messages.length}</span>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span>Turns:</span>
              <span style={{ fontWeight: 600 }}>{Math.ceil(messages.length / 2)}</span>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
