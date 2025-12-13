import { useState, useRef } from 'react';
import axios from 'axios';
import { Upload, Send, FileText, User, Shield, Briefcase, Lock, LogOut } from 'lucide-react';
import './App.css';

const API_URL = "http://localhost:8000";

function App() {
  const [user, setUser] = useState(null); // { role: 'hr' | 'employee', username: string }
  const [activeTab, setActiveTab] = useState('employee');

  if (!user) {
    return <LoginPage onLogin={setUser} />;
  }

  // Handle Tab Switching with Permission Check
  const handleTabSwitch = (tab) => {
    if (tab === 'hr' && user.role !== 'hr') {
      alert("Access Denied: Only HR Administrators can access this portal.");
      return;
    }
    setActiveTab(tab);
  };

  return (
    <div className="container">
      <header>
        <div style={{ maxWidth: 1200, margin: '0 auto', position: 'relative' }}>
          <div style={{ textAlign: 'center' }}>
            <h1>Enterprise GPT</h1>
            <p>Internal Knowledge Base & Support Intelligence</p>
          </div>
          <div style={{ position: 'absolute', right: 0, top: '10%', display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <span style={{ fontWeight: 600, color: '#475569' }}>
              {user.role === 'hr' ? '🛡️ Admin' : '👤 Employee'}
            </span>
            <button
              onClick={() => setUser(null)}
              className="logout-btn"
            >
              <LogOut size={16} style={{ marginRight: 6, display: 'block' }} />
              Logout
            </button>
          </div>
        </div>
      </header>

      <div className="tabs">
        {user.role === 'hr' && (
          <button
            className={`tab-btn hr-tab ${activeTab === 'hr' ? 'active' : ''}`}
            onClick={() => handleTabSwitch('hr')}
          >
            <Shield size={18} style={{ marginRight: 8, verticalAlign: 'text-bottom' }} />
            HR Admin Portal
          </button>
        )}

        <button
          className={`tab-btn employee-tab ${activeTab === 'employee' ? 'active' : ''}`}
          onClick={() => handleTabSwitch('employee')}
        >
          <User size={18} style={{ marginRight: 8, verticalAlign: 'text-bottom' }} />
          Employee Support
        </button>
      </div>

      <div className="card">
        {activeTab === 'hr' ? <HRPortal /> : <EmployeePortal user={user} />}
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
    // Mock Login Logic
    if (username.toLowerCase() === 'admin' && password === 'admin') {
      onLogin({ role: 'hr', username: 'Admin User' });
    } else if (username.toLowerCase() === 'user' && password === 'user') {
      onLogin({ role: 'employee', username: 'Standard User' });
    } else {
      setError('Invalid credentials. Try admin/admin or user/user');
    }
  };

  return (
    <div style={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '100vh',
      background: 'linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%)'
    }}>
      <div style={{
        background: 'white',
        padding: '2.5rem',
        borderRadius: '1rem',
        boxShadow: '0 10px 25px -5px rgba(0,0,0,0.1)',
        width: '100%',
        maxWidth: '400px'
      }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{
            background: '#eff6ff',
            width: 60,
            height: 60,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            margin: '0 auto 1rem',
            color: '#2563eb'
          }}>
            <Lock size={30} />
          </div>
          <h2 style={{
            margin: 0,
            fontFamily: "'Outfit', sans-serif",
            background: "linear-gradient(to right, #1e293b, #2563eb)",
            WebkitBackgroundClip: "text",
            backgroundClip: "text",
            color: "transparent",
            fontWeight: 800
          }}>Enterprise GPT</h2>
          <p style={{ color: '#64748b', margin: '5px 0 0' }}>Sign in to Enterprise GPT</p>
        </div>

        <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div>
            <label style={{ display: 'block', marginBottom: 5, fontWeight: 500, color: '#475569' }}>Username</label>
            <input
              type="text"
              style={{ width: '93%', padding: '.75rem', borderRadius: '.5rem', border: '1px solid #cbd5e1' }}
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="admin or user"
            />
          </div>
          <div>
            <label style={{ display: 'block', marginBottom: 5, fontWeight: 500, color: '#475569' }}>Password</label>
            <input
              type="password"
              style={{ width: '93%', padding: '.75rem', borderRadius: '.5rem', border: '1px solid #cbd5e1' }}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="admin or user"
            />
          </div>

          {error && <p style={{ color: '#ef4444', fontSize: '0.9rem', margin: 0 }}>{error}</p>}

          <button
            type="submit"
            className="btn-primary"
            style={{ marginTop: '0.5rem' }}
          >
            Sign In
          </button>

          <div style={{ marginTop: '1rem', fontSize: '0.85rem', color: '#94a3b8', textAlign: 'center' }}>
            <p>Demo Credentials:</p>
            HR Admin: admin / admin<br />
            Employee: user / user
          </div>
        </form>
      </div>
    </div>
  );
}

function HRPortal() {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'Hello Admin. I am ready to answer questions about candidate resumes or internal HR policies.' }
  ]);

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files));
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    setUploading(true);
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    try {
      const res = await axios.post(`${API_URL}/process-documents`, formData);
      alert(res.data.message);
      setFiles([]);
    } catch (err) {
      alert("Upload failed");
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <>
      <div className="sidebar">
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Briefcase size={20} /> Knowledge Base
        </h3>
        <p style={{ fontSize: '0.9rem', color: '#64748b' }}>Upload resumes (PDF/DOCX) to update the vector database.</p>

        <div className="upload-zone">
          <input
            type="file"
            multiple
            onChange={handleFileChange}
            style={{ display: 'none' }}
            id="file-upload"
          />
          <label htmlFor="file-upload" style={{ cursor: 'pointer', display: 'block' }}>
            <Upload size={32} color="#94a3b8" />
            <p style={{ margin: '10px 0 0', fontWeight: 500 }}>Click to select files</p>
          </label>
        </div>

        {files.length > 0 && (
          <div style={{ fontSize: '0.85rem' }}>
            <strong>Selected:</strong>
            <ul style={{ paddingLeft: 20, margin: '5px 0' }}>
              {files.map((f, i) => <li key={i}>{f.name}</li>)}
            </ul>
          </div>
        )}

        <button
          className="btn-primary"
          onClick={handleUpload}
          disabled={uploading || files.length === 0}
          style={{ opacity: (uploading || files.length === 0) ? 0.7 : 1 }}
        >
          {uploading ? 'Processing...' : 'Upload Documents'}
        </button>
      </div>

      <ChatInterface
        messages={messages}
        setMessages={setMessages}
        portal="hr"
        placeholder="Ask about a specific candidate..."
      />
    </>
  );
}

function EmployeePortal() {
  const [messages, setMessages] = useState([
    { role: 'bot', text: 'Hi! I can help you with leave policies, benefits, and office guidelines. What do you need help with?' }
  ]);

  return (
    <>
      <div className="sidebar">
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <User size={20} /> Employee Help
        </h3>
        <p style={{ fontSize: '0.9rem', color: '#64748b' }}>Common Topics:</p>
        <ul style={{ paddingLeft: 20, lineHeight: 1.8 }}>
          <li>🏖️ Leave Policy</li>
          <li>🏥 Health Insurance</li>
          <li>💰 Payroll Dates</li>
          <li>🏠 Remote Work</li>
        </ul>
      </div>

      <ChatInterface
        messages={messages}
        setMessages={setMessages}
        portal="employee"
        placeholder="How do I apply for leave?"
      />
    </>
  );
}

function ChatInterface({ messages, setMessages, portal, placeholder }) {
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMsg = { role: 'user', text: input };
    setMessages(prev => [...prev, newMsg]);
    setInput('');
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('message', input);
      formData.append('portal', portal);

      const res = await axios.post(`${API_URL}/chat`, formData);
      setMessages(prev => [...prev, { role: 'bot', text: res.data.response }]);
    } catch (err) {
      setMessages(prev => [...prev, { role: 'bot', text: "Error connecting to server." }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="main-content">
      <div className="chat-area">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            {msg.text}
          </div>
        ))}
        {loading && <div className="message bot">Thinking...</div>}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        <input
          className="chat-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          placeholder={placeholder}
        />
        <button
          onClick={sendMessage}
          style={{
            background: 'var(--primary)',
            border: 'none',
            width: 50,
            borderRadius: '0.5rem',
            color: 'white',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <Send size={20} />
        </button>
      </div>
    </div>
  )
}

export default App;
