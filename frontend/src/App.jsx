import { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Send, User, Shield, LogOut, Plus, MessageSquare,
  Paperclip, RefreshCw, Briefcase
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

//const API_URL = "http://localhost:8000";
const API_URL = "/api";

function App() {
  const [user, setUser] = useState(null); // { role: 'hr' | 'employee', username: string }
  const [activeTab, setActiveTab] = useState('employee');
  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);

  // Initialize a default chat when logging in or when no chats exist
  useEffect(() => {
    if (user && chats.length === 0) {
      createNewChat(user.role === 'hr' ? 'hr' : 'employee');
    }
  }, [user]);

  // Handle Tab Switching
  const handleTabSwitch = (tab) => {
    if (tab === 'hr' && user.role !== 'hr') {
      alert("Access Denied");
      return;
    }
    setActiveTab(tab);
  };

  const createNewChat = (type) => {
    const newChat = {
      id: Date.now(),
      title: 'New Chat',
      type: type,
      messages: [{
        role: 'bot',
        text: type === 'hr'
          ? 'Hello Admin. Ask me about candidate resumes or policies.'
          : 'Hi! How can I help you with leave, benefits, or office policies?'
      }],
      timestamp: new Date()
    };
    setChats(prev => [newChat, ...prev]);
    setActiveChatId(newChat.id);
  };

  const updateChatMessages = (chatId, newMsg) => {
    setChats(prev => prev.map(chat => {
      if (chat.id === chatId) {
        // Update title based on first user message if it's "New Chat"
        let title = chat.title;
        if (title === 'New Chat' && newMsg.role === 'user') {
          title = newMsg.text.slice(0, 30) + (newMsg.text.length > 30 ? '...' : '');
        }
        return { ...chat, messages: [...chat.messages, newMsg], title };
      }
      return chat;
    }));
  };

  const handleRestartChat = (chatId) => {
    setChats(prev => prev.map(chat => {
      if (chat.id === chatId) {
        return {
          ...chat,
          messages: [{
            role: 'bot',
            text: chat.type === 'hr'
              ? 'Hello Admin. Ask me about candidate resumes or policies.'
              : 'Hi! How can I help you with leave, benefits, or office policies?'
          }]
        };
      }
      return chat;
    }));
  };

  const activeChat = chats.find(c => c.id === activeChatId) || chats[0];

  const fileInputRef = useRef(null);

  // File Upload Logic for HR (Moved to App for Sidebar access)
  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    if (!activeChatId) return;

    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    // Optimistic update using updateChatMessages directly
    // checking if we have a valid chat first
    const currentChat = chats.find(c => c.id === activeChatId);
    if (!currentChat) return;

    updateChatMessages(activeChatId, { role: 'user', text: `Uploaded ${files.length} document(s): ${files.map(f => f.name).join(', ')}` });

    try {
      const res = await axios.post(`${API_URL}/process-documents`, formData);
      updateChatMessages(activeChatId, { role: 'bot', text: `✅ ${res.data.message}` });
    } catch (err) {
      updateChatMessages(activeChatId, { role: 'bot', text: "❌ Upload failed." });
    }

    // Reset input
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Filter chats based on active tab for Admin, or just show all for Employee (who only has one type)
  // Actually, for Admin, showing both is confusing if tabs are "filters".
  // Let's assume tabs toggle the "Main View" context.
  // The sidebar list should probably filter by the Active Tab context to keep it clean.
  const filteredChats = chats.filter(c => {
    if (!user) return false;
    if (user.role === 'employee') return true;
    return c.type === activeTab;
  });

  const handleLogout = () => {
    setUser(null);
    setChats([]);
    setActiveChatId(null);
  };

  if (!user) {
    return <LoginPage onLogin={(u) => {
      setUser(u);
      setActiveTab(u.role === 'hr' ? 'hr' : 'employee');
    }} />;
  }

  return (
    <div className="app-root">
      {/* GLOBAL HEADER - Full Screen Width */}
      <header className="global-header">
        <h1 className="global-title">Enterprise GPT</h1>
        {activeChatId && (
          <button
            onClick={() => handleRestartChat(activeChatId)}
            style={{
              position: 'absolute',
              right: '2rem',
              background: 'transparent',
              border: '1px solid #e2e8f0',
              cursor: 'pointer',
              color: '#64748b',
              padding: '0.5rem',
              borderRadius: '0.5rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s'
            }}
            title="Restart Chat"
          >
            <RefreshCw size={18} />
          </button>
        )}
      </header>

      <div className="app-layout">
        {/* LEFT PANEL - SIDEBAR (25%) */}
        <div className="left-panel">
          {/* HR Portal Header (New Request) */}
          {user.role === 'hr' && (
            <div className="portal-header-container">
              <div className="portal-icon-square">HR</div>
              <div>
                <div className="portal-title-text">HR Portal</div>
                <div className="portal-subtitle-text">Authorised Access Only</div>
              </div>
            </div>
          )}

          {/* Employee Portal Header - Keeping consistent style or distinct?
              User specifically requested changes for "HR Portal".
              I'll keep Employee as a simpler header for now or match style.
              Let's keep Employee as tab style for now to strictly follow "New HR portal..."
              unless it looks terrible.
              Actually, mixing styles is bad. Let's make Employee look decent too,
              but maybe without the big square if not asked?
              Let's stick to the prompt: modifies HR Portal.
          */}
          {user.role === 'employee' && (
            <div className="portal-header-container">
              <div className="portal-icon-square">EM</div>
              <div>
                <div className="portal-title-text">Employee Portal</div>
                <div className="portal-subtitle-text">Authorised Access Only</div>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="sidebar-actions">
            <button
              className="new-chat-btn"
              onClick={() => createNewChat(activeTab)}
            >
              <img
                src="https://cdn-icons-png.flaticon.com/512/1237/1237946.png"
                alt="New Chat"
                style={{ width: 24, height: 24, opacity: 0.8 }}
              />
              New Chat
            </button>

          </div>

          <div className="chat-history-header">Your chats</div>

          {/* Chat List */}
          <div className="chat-list">
            {filteredChats.map(chat => (
              <div
                key={chat.id}
                className={`chat-item ${activeChatId === chat.id ? 'active' : ''}`}
                onClick={() => setActiveChatId(chat.id)}
              >
                <MessageSquare size={18} />
                <span className="chat-item-title">{chat.title}</span>
              </div>
            ))}
            {filteredChats.length === 0 && (
              <div style={{ padding: '1rem', textAlign: 'center', color: '#94a3b8', fontSize: '0.9rem' }}>
                No chats yet.
              </div>
            )}
          </div>

          {/* Logout Button (Bottom) */}
          <div className="sidebar-footer">
            <button className="logout-full-btn" onClick={handleLogout}>
              <LogOut size={18} />
              Logout
            </button>
          </div>
        </div>

        {/* RIGHT PANEL - CHAT INTERFACE (75%) */}
        <div className="right-panel">
          {activeChat ? (
            <ChatInterface
              chat={activeChat}
              onUpdateMessages={updateChatMessages}
              user={user}
            />
          ) : (
            <div className="empty-state">
              <img
                src="https://cdn-icons-png.flaticon.com/2040/2040504.png"
                width="64" alt="Chat" style={{ opacity: 0.2, marginBottom: '1rem' }}
              />
              <h3>Select a chat to start messaging</h3>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function ChatInterface({ chat, onUpdateMessages, user }) {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chat.messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMsg = { role: 'user', text: input };
    onUpdateMessages(chat.id, newMsg);
    setInput('');
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('message', input);
      formData.append('portal', chat.type);

      const res = await axios.post(`${API_URL}/chat`, formData);
      onUpdateMessages(chat.id, { role: 'bot', text: res.data.response });
    } catch (err) {
      onUpdateMessages(chat.id, { role: 'bot', text: "Error connecting to server." });
    } finally {
      setLoading(false);
    }
  };

  // File Upload Logic for HR
  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    // Optimistic update
    onUpdateMessages(chat.id, { role: 'user', text: `Uploaded ${files.length} document(s): ${files.map(f => f.name).join(', ')}` });
    setLoading(true);

    try {
      const res = await axios.post(`${API_URL}/process-documents`, formData);
      onUpdateMessages(chat.id, { role: 'bot', text: `✅ ${res.data.message}` });
    } catch (err) {
      onUpdateMessages(chat.id, { role: 'bot', text: "❌ Upload failed." });
    } finally {
      setLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  return (
    <div className="chat-container">
      {/* Chat Header removed as requested, actions moved to global header */}

      {/* Messages */}
      <div className="messages-area">
        {chat.messages.map((msg, idx) => (
          <div key={idx} className={`message-bubble ${msg.role}`}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
          </div>
        ))}
        {loading && (
          <div className="message-bubble bot">
            <div className="typing">
              <div className="dot"></div>
              <div className="dot"></div>
              <div className="dot"></div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="input-container">
        <div className="input-wrapper">
          <input
            className="message-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Type a message..."
            disabled={loading}
          />
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {/* Upload Button - Next to Send */}
            {chat.type === 'hr' && user.role === 'hr' && (
              <>
                <input
                  type="file"
                  multiple
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  onChange={handleFileUpload}
                />
                <button
                  className="send-btn"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={loading}
                  title="Upload Document"
                  style={{ background: '#f1f5f9', color: '#64748b' }}
                >
                  <Paperclip size={18} />
                </button>
              </>
            )}

            <button
              className="send-btn"
              onClick={sendMessage}
              disabled={!input.trim() || loading}
            >
              <Send size={18} />
            </button>
          </div>
        </div>
        <div style={{ textAlign: 'center', marginTop: '0.5rem', fontSize: '0.75rem', color: '#94a3b8' }}>
          AI can make mistakes. Verify important information.
        </div>
        {chat.type === 'hr' && (
          <div style={{ textAlign: 'center', marginTop: '0.25rem', fontSize: '0.75rem', color: '#d97706' }}>
            Warning: If resume is being uploaded it should be in word doc and firstname_lastname_date format
          </div>
        )}
      </div>
    </div>
  );
}

function LoginPage({ onLogin }) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleLogin = (e) => {
    e.preventDefault();
    if (username.toLowerCase() === 'admin' && password === 'admin') {
      onLogin({ role: 'hr', username: 'Admin User' });
    } else if (username.toLowerCase() === 'user' && password === 'user') {
      onLogin({ role: 'employee', username: 'Standard User' });
    } else {
      setError('Invalid credentials');
    }
  };

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      width: '100vw',
      background: 'linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%)'
    }}>
      <div style={{
        background: 'white',
        padding: '2.5rem',
        borderRadius: '1.5rem',
        boxShadow: '0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04)',
        width: '100%',
        maxWidth: '400px'
      }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{
            background: '#eff6ff', width: 64, height: 64, borderRadius: '20px',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            margin: '0 auto 1.5rem', color: '#2563eb'
          }}>
            <Briefcase size={32} />
          </div>
          <h2 style={{
            margin: 0, fontFamily: "'Outfit', sans-serif",
            background: "linear-gradient(to right, #1e293b, #2563eb)",
            WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent",
            fontWeight: 800, fontSize: '1.75rem'
          }}>Enterprise GPT</h2>
          <p style={{ color: '#64748b', margin: '0.5rem 0 0' }}>Internal AI Workspace</p>
        </div>

        <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div>
            <label style={{ display: 'block', marginBottom: 6, fontWeight: 500, color: '#475569', fontSize: '0.9rem' }}>Username</label>
            <input
              type="text"
              style={{ width: '100%', padding: '0.75rem', borderRadius: '0.75rem', border: '1px solid #cbd5e1', boxSizing: 'border-box', outline: 'none' }}
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="admin or user"
            />
          </div>
          <div>
            <label style={{ display: 'block', marginBottom: 6, fontWeight: 500, color: '#475569', fontSize: '0.9rem' }}>Password</label>
            <input
              type="password"
              style={{ width: '100%', padding: '0.75rem', borderRadius: '0.75rem', border: '1px solid #cbd5e1', boxSizing: 'border-box', outline: 'none' }}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
            />
          </div>

          {error && <p style={{ color: '#ef4444', fontSize: '0.85rem', margin: 0 }}>{error}</p>}

          <button
            type="submit"
            style={{
              marginTop: '0.5rem', background: '#2563eb', color: 'white',
              border: 'none', padding: '0.875rem', borderRadius: '0.75rem',
              fontWeight: 600, cursor: 'pointer', fontSize: '1rem',
              transition: 'background 0.2s'
            }}
          >
            Sign In
          </button>
        </form>

        <div style={{ marginTop: '1.5rem', padding: '1rem', background: '#f8fafc', borderRadius: '0.75rem', fontSize: '0.8rem', color: '#64748b', textAlign: 'center' }}>
          <div style={{ marginBottom: 4 }}><strong>Admin:</strong> admin / admin</div>
          <div><strong>Employee:</strong> user / user</div>
        </div>
      </div>
    </div>
  );
}

export default App;
