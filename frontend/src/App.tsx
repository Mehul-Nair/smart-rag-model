import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Bot, Sparkles, MessageCircle } from "lucide-react";
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
      content: "Hello! I'm your intelligent home decor assistant. I can help you discover beautiful fabrics, rugs, sofas, curtains, and furniture pieces. What would you like to explore today?",
      sender: "ai",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isInputFocused, setIsInputFocused] = useState(false);
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
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 flex flex-col relative overflow-hidden">
      {/* Background Elements */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50/30 via-transparent to-purple-50/20 pointer-events-none" />
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-gradient-to-br from-blue-200/20 to-purple-200/20 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-gradient-to-br from-purple-200/20 to-pink-200/20 rounded-full blur-3xl pointer-events-none" />

      {/* Header */}
      <motion.header
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="bg-white/70 backdrop-blur-xl border-b border-white/20 shadow-sm relative z-10"
      >
        <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <motion.div
              className="w-12 h-12 bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 rounded-2xl flex items-center justify-center shadow-lg"
              whileHover={{ scale: 1.05, rotate: 5 }}
              transition={{ type: "spring", stiffness: 400, damping: 10 }}
            >
              <Sparkles className="w-6 h-6 text-white drop-shadow-sm" />
            </motion.div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 bg-clip-text text-transparent">
                Decor Intelligence
              </h1>
              <p className="text-sm text-gray-500 font-medium">Your Smart Design Assistant</p>
            </div>
          </div>
          <motion.div
            className="flex items-center space-x-2 text-sm text-gray-500"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            <motion.div
              className="w-2 h-2 bg-green-400 rounded-full shadow-sm"
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            <span className="font-medium">Online</span>
          </motion.div>
        </div>
      </motion.header>

      {/* Chat Container */}
      <div className="flex-1 max-w-4xl mx-auto w-full px-4 py-6 relative z-10">
        <motion.div
          className="bg-white/40 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/30 h-[600px] flex flex-col relative overflow-hidden"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
        >
          {/* Chat container background */}
          <div className="absolute inset-0 bg-gradient-to-b from-white/20 via-white/10 to-white/5 pointer-events-none" />

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6 relative">
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
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
                className="flex items-start space-x-3"
              >
                <motion.div
                  className="w-8 h-8 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg"
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  <Bot className="w-4 h-4 text-white" />
                </motion.div>
                <div className="bg-white/80 backdrop-blur-sm rounded-2xl px-4 py-3 shadow-lg border border-white/50 max-w-xs lg:max-w-md">
                  <TypingIndicator />
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <motion.div
            className="p-6 border-t border-white/30 bg-white/20 backdrop-blur-sm relative"
            animate={{
              backgroundColor: isInputFocused ? "rgba(255, 255, 255, 0.4)" : "rgba(255, 255, 255, 0.2)"
            }}
            transition={{ duration: 0.3 }}
          >
            <div className="flex items-end space-x-4">
              <div className="flex-1">
                <AnimatedTextarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  onFocus={() => setIsInputFocused(true)}
                  onBlur={() => setIsInputFocused(false)}
                  placeholder="Describe your perfect space..."
                  isLoading={isLoading}
                  disabled={isLoading}
                />
              </div>
              <motion.button
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading}
                className={`w-12 h-12 rounded-2xl flex items-center justify-center transition-all duration-300 shadow-lg ${
                  inputValue.trim() && !isLoading
                    ? "bg-gradient-to-br from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-xl hover:shadow-2xl"
                    : "bg-gray-200/80 text-gray-400 cursor-not-allowed backdrop-blur-sm"
                }`}
              >
                <motion.div
                  animate={isLoading ? { rotate: 360 } : { rotate: 0 }}
                  transition={isLoading ? { duration: 1, repeat: Infinity, ease: "linear" } : {}}
                >
                  <Send className="w-5 h-5" />
                </motion.div>
              </motion.button>
            </div>

            <div className="mt-4 flex items-center justify-between">
              <div className="text-xs text-gray-500 font-medium">
                Press Enter to send, Shift+Enter for new line
              </div>
              <motion.button
                onClick={handleTestProductResponse}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="text-xs bg-white/60 hover:bg-white/80 text-gray-600 px-3 py-1.5 rounded-xl transition-all duration-200 backdrop-blur-sm border border-white/50 font-medium shadow-sm"
              >
                Test Product Cards
              </motion.button>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}

export default App;