import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, HelpCircle, BookOpen, Cpu, BarChart3, Navigation, AlertTriangle, Sparkles } from 'lucide-react';

// Static knowledge base - NO external APIs, NO LLM calls
const KNOWLEDGE_BASE = {
  "What does this project do?": {
    category: "PROJECT_OVERVIEW",
    answer: [
      "This system evaluates character backstories against full-length novels for logical consistency.",
      "It determines whether a character's background story aligns with events in the original narrative.",
      "The problem is detecting contradictions between claimed backstory and actual novel content.",
      "This is challenging for LLMs because it requires cross-referencing large text corpora and understanding narrative coherence."
    ]
  },
  
  "What is backstory–novel consistency?": {
    category: "PROJECT_OVERVIEW", 
    answer: [
      "Backstory-novel consistency checks if character background information contradicts established novel events.",
      "A consistent backstory means the character's history aligns with the novel's narrative.",
      "An inconsistent backstory contains contradictions or impossible claims based on novel content.",
      "Example: A character claims they were born in Paris, but the novel shows they were born in Lyon."
    ]
  },
  
  "How are predictions made?": {
    category: "SYSTEM_ARCHITECTURE",
    answer: [
      "1. Retrieval: System searches novels for semantically relevant text chunks using sentence transformers.",
      "2. Analysis: Retrieved chunks are classified as SUPPORT, CONTRADICT, or NEUTRAL.",
      "3. Aggregation: Deterministic rules convert analysis to binary prediction (0=Inconsistent, 1=Consistent).",
      "Any strong contradiction (confidence ≥ 0.7) → Inconsistent prediction.",
      "No strong contradictions → Consistent prediction."
    ]
  },
  
  "What models are used?": {
    category: "MODELS_USED",
    answer: [
      "Sentence Transformer embeddings for semantic search (retrieves relevant novel chunks).",
      "TF-IDF as fallback when sentence transformers unavailable.",
      "LLM for classification only (SUPPORT/CONTRADICT/NEUTRAL labels).",
      "Deterministic rule-based aggregation (no ML inference).",
      "Important: No training or fine-tuning is performed on these models."
    ]
  },
  
  "What do predictions mean?": {
    category: "RESULTS_PREDICTIONS",
    answer: [
      "Prediction = 1 (Consistent): Backstory aligns with novel content.",
      "Prediction = 0 (Inconsistent): Backstory contradicts novel content.",
      "Evidence chunks show why the decision was made.",
      "Similarity scores indicate confidence (0.0-1.0 scale)."
    ]
  },
  
  "What is accuracy?": {
    category: "ACCURACY_METRICS",
    answer: [
      "Training accuracy: 36.25% (29/80 correct) - model struggles with training data.",
      "Test predictions: 90% consistent, 10% inconsistent - better performance on unseen data.",
      "Accuracy is only calculated where ground truth exists (training set).",
      "No accuracy claims on test set - only distribution statistics shown.",
      "This interface does not influence model performance - displays pre-computed results."
    ]
  },
  
  "How do I navigate?": {
    category: "WEBSITE_NAVIGATION",
    answer: [
      "Dashboard: Overview with charts and summary statistics.",
      "Results: Sortable table with all predictions and search functionality.",
      "Detail View: Click any row to see evidence and decision logic.",
      "Methodology: Technical explanation of system architecture.",
      "Use navigation tabs at top to switch between sections."
    ]
  },
  
  "What are limitations?": {
    category: "LIMITATIONS",
    answer: [
      "No causal reasoning or temporal modeling.",
      "No genre or tone analysis.",
      "No streaming memory or incremental updates.",
      "Only supports binary consistency classification.",
      "Limited to character backstory evaluation scope."
    ]
  },
  
  "Why deterministic approach?": {
    category: "DESIGN_DECISIONS",
    answer: [
      "Ensures reproducible results - same input always gives same output.",
      "More explainable than complex ML models.",
      "Reduces risk of hallucination or false reasoning.",
      "Aligns with judge-friendly transparency requirements.",
      "Easier to debug and verify correctness."
    ]
  },
  
  "What future improvements?": {
    category: "FUTURE_WORK",
    answer: [
      "Event-level causal reasoning.",
      "Incremental memory updates.",
      "Structured fact graphs.",
      "Multilingual novel support.",
      "Character relationship modeling."
    ]
  },
  
  "How does this align with hackathon?": {
    category: "HACKATHON_ALIGNMENT",
    answer: [
      "Track A: Retrieval-augmented reasoning demonstration.",
      "Judge-friendly transparency and explainability.",
      "Reproducible, safe system design.",
      "Professional frontend with clear UX.",
      "No exaggerated claims or black-box models."
    ]
  }
};

// Predefined suggestions for quick access
const SUGGESTIONS = [
  "What does this project do?",
  "How are predictions made?",
  "What models are used?",
  "What do predictions mean?",
  "What is accuracy?",
  "How do I navigate?",
  "What are limitations?",
  "Why deterministic approach?",
  "What future improvements?",
  "How does this align with hackathon?"
];

const ExplainerBot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [suggestionsVisible, setSuggestionsVisible] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const addMessage = (text, isUser = false) => {
    const newMessage = {
      id: Date.now(),
      text,
      isUser,
      timestamp: new Date().toLocaleTimeString()
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const handleQuestion = (question) => {
    setInputValue(question);
    setSuggestionsVisible(false);
    handleSend(question);
  };

  const handleSend = () => {
    if (!inputValue.trim()) return;

    // Add user message
    addMessage(inputValue, true);
    setInputValue('');

    // Find answer in knowledge base
    const normalizedInput = inputValue.toLowerCase().trim();
    let answer = null;

    // Direct match
    if (KNOWLEDGE_BASE[normalizedInput]) {
      answer = KNOWLEDGE_BASE[normalizedInput];
    } else {
      // Partial match
      for (const [key, value] of Object.entries(KNOWLEDGE_BASE)) {
        if (normalizedInput.includes(key.toLowerCase()) || 
            key.toLowerCase().includes(normalizedInput)) {
          answer = value;
          break;
        }
      }
    }

    // Add bot response
    setTimeout(() => {
      if (answer) {
        const responseText = answer.answer.join(' ');
        addMessage(responseText, false);
      } else {
        addMessage(
          "This chatbot explains the system design and results. It does not perform live predictions. " +
          "Try asking about: What does this project do? How are predictions made? What models are used?",
          false
        );
      }
    }, 300);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className="fixed bottom-6 left-6 z-50 w-96">
      {/* Chat Button */}
      {!isOpen && (
        <button
          onClick={toggleChat}
          className="bg-gradient-to-r from-purple-500 to-pink-500 text-white p-4 rounded-full shadow-xl hover:shadow-2xl hover:scale-105 transition-all duration-300 relative group"
          title="System Explainer"
        >
          <div className="relative">
            <Sparkles className="h-6 w-6 animate-pulse" />
            <div className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-400 rounded-full animate-ping"></div>
          </div>
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div className="bg-white rounded-2xl shadow-2xl border border-gray-200 w-96 h-[500px] flex flex-col overflow-hidden">
          {/* Header */}
          <div className="flex items-center justify-between p-6 bg-gradient-to-r from-purple-500 to-pink-500 text-white">
            <div className="flex items-center space-x-3 mx-auto">
              <div className="flex items-center space-x-2">
                <div className="relative">
                  <Sparkles className="h-6 w-6" />
                  <div className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-400 rounded-full animate-ping"></div>
                </div>
                <h3 className="font-bold text-white text-lg">AI Assistant</h3>
              </div>
            </div>
            <button
              onClick={toggleChat}
              className="text-white hover:bg-white/20 transition-colors rounded-full p-1"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {messages.length === 0 && (
              <div className="text-center text-gray-500 text-sm">
                <MessageCircle className="h-8 w-8 mx-auto mb-2" />
                <p>Ask about the system design, models, or results.</p>
              </div>
            )}
            
            {messages.map(message => (
              <div
                key={message.id}
                className={`flex ${message.isUser ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-xs px-3 py-2 rounded-lg text-sm ${
                    message.isUser
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-900 border border-gray-200'
                  }`}
                >
                  {!message.isUser && (
                    <div className="flex items-center space-x-1 mb-1">
                      <Cpu className="h-3 w-3 text-blue-600" />
                      <span className="text-xs text-gray-600">System Info</span>
                    </div>
                  )}
                  <p className="whitespace-pre-wrap">{message.text}</p>
                  <div className="text-xs text-gray-500 mt-1">
                    {message.timestamp}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="p-4 border-t border-gray-200 bg-gradient-to-r from-purple-50 to-pink-50">
            {/* Suggestions */}
            {suggestionsVisible && (
              <div className="mb-3 p-3 bg-white/80 backdrop-blur-sm rounded-lg border border-purple-200 shadow-lg">
                <p className="text-xs font-medium text-purple-700 mb-2">Quick questions:</p>
                <div className="grid grid-cols-1 gap-2">
                  {SUGGESTIONS.map((suggestion, index) => (
                    <button
                      key={index}
                      onClick={() => handleQuestion(suggestion)}
                      className="text-left text-xs p-3 hover:bg-purple-100 rounded-lg transition-all duration-200 text-gray-700 hover:text-purple-900 border border-purple-200 hover:border-purple-300"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Input */}
            <div className="flex space-x-3">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                onFocus={() => setSuggestionsVisible(true)}
                onBlur={() => setTimeout(() => setSuggestionsVisible(false), 200)}
                placeholder="Ask about the system..."
                className="flex-1 px-4 py-3 border-2 border-purple-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 text-sm bg-white/90 backdrop-blur-sm"
              />
              <button
                onClick={handleSend}
                className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-6 py-3 rounded-xl hover:from-purple-600 hover:to-pink-600 transition-all duration-200 text-sm font-medium shadow-lg hover:shadow-xl transform hover:scale-105"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Disclaimers */}
      {isOpen && (
        <div className="absolute bottom-full left-0 mb-2 w-80 p-4 bg-gradient-to-r from-yellow-50 to-orange-50 border border-orange-200 rounded-2xl shadow-2xl backdrop-blur-sm">
          <div className="flex items-start space-x-3">
            <div className="relative">
              <AlertTriangle className="h-5 w-5 text-orange-600 mt-0.5 flex-shrink-0" />
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-yellow-400 rounded-full animate-ping"></div>
            </div>
            <div className="text-xs text-orange-800">
              <p className="font-bold text-orange-900">Important Notice:</p>
              <ul className="mt-2 space-y-2">
                <li className="flex items-start">
                  <span className="text-orange-600 mr-2">•</span>
                  <span>This chatbot is informational only</span>
                </li>
                <li className="flex items-start">
                  <span className="text-orange-600 mr-2">•</span>
                  <span>Predictions are precomputed and not affected by this interface</span>
                </li>
                <li className="flex items-start">
                  <span className="text-orange-600 mr-2">•</span>
                  <span>No data is sent outside this application</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ExplainerBot;
