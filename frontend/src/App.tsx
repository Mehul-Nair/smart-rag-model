import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Bot, Sparkles } from "lucide-react";
import ChatMessage from "./components/ChatMessage";
import AnimatedTextarea from "./components/AnimatedTextarea";
import TypingIndicator from "./components/TypingIndicator";

interface Product {
  name: string;
  price: string;
  url: string;
}

interface ProductResponseData {
  type: string;
  summary: string;
  products: Product[];
}

interface CategoryNotFoundData {
  type: string;
  requested_category: string;
  available_categories: string[];
  message: string;
}

interface BudgetConstraintData {
  type: string;
  category: string;
  requested_budget: string;
  message: string;
}

interface Message {
  id: string;
  content:
    | string
    | ProductResponseData
    | CategoryNotFoundData
    | BudgetConstraintData
    | string[];
  sender: "user" | "ai";
  timestamp: Date;
  userQuery?: string;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content:
        "Hello! I'm your AI assistant for home decor products. I can help you find fabrics, rugs, sofas, curtains, and other furniture items. What are you looking for today?",
      sender: "ai",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue.trim(),
      sender: "user",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);

    try {
      const response = await fetch("/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: userMessage.content,
          session_id: "default",
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response");
      }

      const data = await response.json();
      console.log("Backend response:", data); // Debug log

      // Handle the new response format from backend
      let parsedContent:
        | string
        | ProductResponseData
        | CategoryNotFoundData
        | BudgetConstraintData
        | string[];
      if (
        data.type === "product_suggestion" &&
        typeof data.response === "object"
      ) {
        parsedContent = data.response as ProductResponseData;
      } else if (
        data.type === "category_not_found" &&
        typeof data.response === "object"
      ) {
        parsedContent = data.response as CategoryNotFoundData;
      } else if (
        data.type === "budget_constraint" &&
        typeof data.response === "object"
      ) {
        parsedContent = data.response as BudgetConstraintData;
      } else if (
        data.type === "category_list" &&
        Array.isArray(data.response)
      ) {
        parsedContent = data.response as string[];
      } else {
        parsedContent = data.response as string;
      }

      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: parsedContent,
        sender: "ai",
        timestamp: new Date(),
        userQuery:
          (data.type === "product_suggestion" ||
            data.type === "budget_constraint") &&
          typeof userMessage.content === "string"
            ? userMessage.content
            : undefined,
      };

      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content:
          "Sorry, I'm having trouble connecting right now. Please try again in a moment.",
        sender: "ai",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Handle category click
  const handleCategoryClick = (category: string) => {
    setInputValue(category);
  };

  // Test function to simulate product response
  const handleTestProductResponse = () => {
    const testProductResponse: ProductResponseData = {
      type: "product_suggestion",
      summary:
        "Here are some beautiful home decor products that match your request:",
      products: [
        {
          name: "Modern Velvet Sofa",
          price: "$1,299",
          url: "https://example.com/sofa1",
        },
        {
          name: "Persian Area Rug",
          price: "$899",
          url: "https://example.com/rug1",
        },
        {
          name: "Elegant Curtain Set",
          price: "$299",
          url: "https://example.com/curtain1",
        },
      ],
    };

    const testMessage: Message = {
      id: Date.now().toString(),
      content: testProductResponse,
      sender: "ai",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, testMessage]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex flex-col">
      {/* Header */}
      <motion.header
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        className="bg-white/80 backdrop-blur-md border-b border-gray-200/50 shadow-sm"
      >
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-r from-primary-500 to-primary-600 rounded-xl flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">
                Smart AI Agent
              </h1>
              <p className="text-sm text-gray-600">Home Decor Assistant</p>
            </div>
          </div>
          <div className="flex items-center space-x-2 text-sm text-gray-500">
            <div className="w-2 h-2 bg-green-400 rounded-full"></div>
            <span>Online</span>
          </div>
        </div>
      </motion.header>

      {/* Chat Container */}
      <div className="flex-1 max-w-4xl mx-auto w-full px-4 py-6">
        <div className="bg-white/60 backdrop-blur-sm rounded-2xl shadow-xl border border-white/20 h-[600px] flex flex-col">
          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            <AnimatePresence>
              {messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                  onCategoryClick={handleCategoryClick}
                />
              ))}
            </AnimatePresence>

            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-start space-x-3"
              >
                <div className="w-8 h-8 bg-gradient-to-r from-primary-400 to-primary-500 rounded-full flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="bg-white rounded-2xl px-4 py-3 shadow-sm border border-gray-100 max-w-xs lg:max-w-md">
                  <TypingIndicator />
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-6 border-t border-gray-100/50">
            <div className="flex items-end space-x-3">
              <div className="flex-1">
                <AnimatedTextarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask me about home decor products..."
                  isLoading={isLoading}
                  disabled={isLoading}
                />
              </div>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading}
                className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all duration-200 ${
                  inputValue.trim() && !isLoading
                    ? "bg-primary-500 hover:bg-primary-600 text-white shadow-lg hover:shadow-xl"
                    : "bg-gray-200 text-gray-400 cursor-not-allowed"
                }`}
              >
                <Send className="w-5 h-5" />
              </motion.button>
            </div>

            <div className="mt-3 flex items-center justify-between">
              <div className="text-xs text-gray-500">
                Press Enter to send, Shift+Enter for new line
              </div>
              <button
                onClick={handleTestProductResponse}
                className="text-xs bg-gray-100 hover:bg-gray-200 text-gray-600 px-3 py-1 rounded-lg transition-colors duration-200"
              >
                Test Product Cards
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
