import React, { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Bot, Sparkles, MessageCircle } from "lucide-react";
import ChatMessage from "./components/ChatMessage";
import AnimatedTextarea from "./components/AnimatedTextarea";
import TypingIndicator from "./components/TypingIndicator";
import DarkModeToggle from "./components/DarkModeToggle";
import { useDarkMode } from "./hooks/useDarkMode";

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

interface CategoryListData {
  type: string;
  categories: string[];
  message: string;
}

interface Message {
  id: string;
  content:
    | string
    | ProductResponseData
    | CategoryNotFoundData
    | BudgetConstraintData
    | CategoryListData
    | string[];
  sender: "user" | "ai";
  timestamp: Date;
  userQuery?: string;
}

function App() {
  const { isDark, toggleTheme, mounted } = useDarkMode();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content:
        "Hello! I'm your intelligent home decor assistant. I can help you discover beautiful fabrics, rugs, sofas, curtains, and furniture pieces. What would you like to explore today?",
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
        | CategoryListData
        | string[];

      console.log("Response type:", data.type);
      console.log("Response data:", data.response);

      if (
        data.type === "product_suggestion" &&
        typeof data.response === "object"
      ) {
        console.log("Processing product_suggestion");
        parsedContent = data.response as ProductResponseData;
      } else if (
        data.type === "category_not_found" &&
        typeof data.response === "object"
      ) {
        console.log("Processing category_not_found");
        parsedContent = data.response as CategoryNotFoundData;
      } else if (
        data.type === "budget_constraint" &&
        typeof data.response === "object"
      ) {
        console.log("Processing budget_constraint");
        parsedContent = data.response as BudgetConstraintData;
      } else if (
        data.type === "category_list" &&
        typeof data.response === "object" &&
        data.response.categories
      ) {
        console.log("Processing category_list");
        parsedContent = data.response.categories as string[];
      } else if (
        data.type === "clarification" &&
        typeof data.response === "object" &&
        data.response.message
      ) {
        console.log("Processing clarification");
        parsedContent = data.response.message;
      } else if (
        data.type === "greeting" &&
        typeof data.response === "object"
      ) {
        console.log("Processing greeting");
        parsedContent = data.response.message;
      } else if (data.type === "error") {
        // Handle error type from backend
        if (typeof data.response === "object" && data.response.message) {
          parsedContent = data.response.message;
        } else if (typeof data.response === "string") {
          parsedContent = data.response;
        } else {
          parsedContent = "An unknown error occurred.";
        }
      } else {
        console.log("Processing as string response");
        parsedContent = data.response as string;
      }

      console.log("Final parsed content:", parsedContent);

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
          price: "12999",
          url: "https://example.com/sofa1",
        },
        {
          name: "Persian Area Rug",
          price: "8999",
          url: "https://example.com/rug1",
        },
        {
          name: "Elegant Curtain Set",
          price: "2999",
          url: "https://example.com/curtain1",
        },
        {
          name: "Alison Bed Side Table",
          price: "13498",
          url: "https://example.com/bedside1",
        },
        {
          name: "Marcus Bedside Table",
          price: "3599",
          url: "https://example.com/bedside2",
        },
        {
          name: "Bonin Bedside Table",
          price: "5999",
          url: "https://example.com/bedside3",
        },
        {
          name: "Luxury Fabric Collection",
          price: "2499",
          url: "https://example.com/fabric1",
        },
        {
          name: "Handmade Wool Rug",
          price: "15999",
          url: "https://example.com/rug2",
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

  // Test function to simulate category not found response
  const handleTestCategoryNotFound = () => {
    const testCategoryNotFound: CategoryNotFoundData = {
      type: "category_not_found",
      requested_category: "dining chairs",
      available_categories: ["bedside tables", "sofas", "curtains", "rugs"],
      message:
        "We don't have dining chairs in our catalog. Here are our available categories.",
    };

    const testMessage: Message = {
      id: Date.now().toString(),
      content: testCategoryNotFound,
      sender: "ai",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, testMessage]);
  };

  // Test function to simulate budget constraint response
  const handleTestBudgetConstraint = () => {
    const testBudgetConstraint: BudgetConstraintData = {
      type: "budget_constraint",
      category: "sofas",
      requested_budget: "under 500",
      message: "We found sofas but none under your budget of under 500.",
    };

    const testMessage: Message = {
      id: Date.now().toString(),
      content: testBudgetConstraint,
      sender: "ai",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, testMessage]);
  };

  // Loading button function to simulate API call with animated border
  const handleLoadingButton = async () => {
    if (isLoading) return;

    setIsLoading(true);

    // Simulate API call delay
    await new Promise((resolve) => setTimeout(resolve, 10000));

    // Add a simple response after loading
    const loadingResponse: Message = {
      id: Date.now().toString(),
      content:
        "This is a simulated response from the loading button. The animated border effect has been tested successfully! 🎉",
      sender: "ai",
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, loadingResponse]);
    setIsLoading(false);
  };

  // Don't render until mounted to prevent hydration mismatch
  if (!mounted) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 flex flex-col relative overflow-hidden transition-colors duration-300">
      {/* Background Elements */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50/30 via-transparent to-purple-50/20 dark:from-blue-900/20 dark:via-transparent dark:to-purple-900/20 pointer-events-none" />

      {/* Animated gradient orbs */}
      <motion.div
        className="absolute top-0 left-1/4 w-96 h-96 rounded-full blur-3xl pointer-events-none"
        animate={{
          background: [
            "radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.1) 50%, transparent 100%)",
            "radial-gradient(circle, rgba(147, 51, 234, 0.1) 0%, rgba(236, 72, 153, 0.1) 50%, transparent 100%)",
            "radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, rgba(147, 51, 234, 0.1) 50%, transparent 100%)",
          ],
        }}
        transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
      />

      <motion.div
        className="absolute bottom-0 right-1/4 w-96 h-96 rounded-full blur-3xl pointer-events-none"
        animate={{
          background: [
            "radial-gradient(circle, rgba(147, 51, 234, 0.1) 0%, rgba(236, 72, 153, 0.1) 50%, transparent 100%)",
            "radial-gradient(circle, rgba(236, 72, 153, 0.1) 0%, rgba(59, 130, 246, 0.1) 50%, transparent 100%)",
            "radial-gradient(circle, rgba(147, 51, 234, 0.1) 0%, rgba(236, 72, 153, 0.1) 50%, transparent 100%)",
          ],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 2,
        }}
      />

      {/* Dark mode animated orbs */}
      <motion.div
        className="absolute top-0 left-1/4 w-96 h-96 rounded-full blur-3xl pointer-events-none dark:block hidden"
        animate={{
          background: [
            "radial-gradient(circle, rgba(59, 130, 246, 0.2) 0%, rgba(147, 51, 234, 0.2) 50%, transparent 100%)",
            "radial-gradient(circle, rgba(147, 51, 234, 0.2) 0%, rgba(236, 72, 153, 0.2) 50%, transparent 100%)",
            "radial-gradient(circle, rgba(59, 130, 246, 0.2) 0%, rgba(147, 51, 234, 0.2) 50%, transparent 100%)",
          ],
        }}
        transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
      />

      <motion.div
        className="absolute bottom-0 right-1/4 w-96 h-96 rounded-full blur-3xl pointer-events-none dark:block hidden"
        animate={{
          background: [
            "radial-gradient(circle, rgba(147, 51, 234, 0.2) 0%, rgba(236, 72, 153, 0.2) 50%, transparent 100%)",
            "radial-gradient(circle, rgba(236, 72, 153, 0.2) 0%, rgba(59, 130, 246, 0.2) 50%, transparent 100%)",
            "radial-gradient(circle, rgba(147, 51, 234, 0.2) 0%, rgba(236, 72, 153, 0.2) 50%, transparent 100%)",
          ],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 2,
        }}
      />

      {/* Additional floating elements */}
      <motion.div
        className="absolute top-1/3 right-1/3 w-64 h-64 rounded-full blur-2xl pointer-events-none opacity-30"
        animate={{
          y: [0, -20, 0],
          x: [0, 10, 0],
          scale: [1, 1.1, 1],
          background: [
            "radial-gradient(circle, rgba(59, 130, 246, 0.05) 0%, transparent 70%)",
            "radial-gradient(circle, rgba(147, 51, 234, 0.05) 0%, transparent 70%)",
            "radial-gradient(circle, rgba(59, 130, 246, 0.05) 0%, transparent 70%)",
          ],
        }}
        transition={{
          duration: 12,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 1,
        }}
      />

      <motion.div
        className="absolute bottom-1/3 left-1/3 w-48 h-48 rounded-full blur-2xl pointer-events-none opacity-20 dark:opacity-30"
        animate={{
          y: [0, 15, 0],
          x: [0, -8, 0],
          scale: [1, 0.9, 1],
          background: [
            "radial-gradient(circle, rgba(236, 72, 153, 0.05) 0%, transparent 70%)",
            "radial-gradient(circle, rgba(59, 130, 246, 0.05) 0%, transparent 70%)",
            "radial-gradient(circle, rgba(236, 72, 153, 0.05) 0%, transparent 70%)",
          ],
        }}
        transition={{
          duration: 15,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 3,
        }}
      />

      {/* Header */}
      <motion.header
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
        className="bg-white/70 dark:bg-gray-800/70 backdrop-blur-xl border-b border-white/20 dark:border-gray-700/20 shadow-sm relative z-10"
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
              <h1 className="text-xl font-bold bg-gradient-to-r from-gray-900 to-gray-700 dark:from-gray-100 dark:to-gray-300 bg-clip-text text-transparent">
                Decor Intelligence
              </h1>
              <p className="text-sm text-gray-500 dark:text-gray-400 font-medium">
                Your Smart Design Assistant
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <DarkModeToggle isDark={isDark} onToggle={toggleTheme} />
            <motion.div
              className="flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400"
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
        </div>
      </motion.header>

      {/* Chat Container */}
      <div className="flex-1 max-w-4xl mx-auto w-full px-4 py-6 relative z-10">
        <motion.div
          className="bg-white/40 dark:bg-gray-800/40 backdrop-blur-xl rounded-3xl shadow-2xl border border-white/30 dark:border-gray-700/30 h-[600px] flex flex-col relative overflow-hidden"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
        >
          {/* Chat container background */}
          <div className="absolute inset-0 bg-gradient-to-b from-white/20 via-white/10 to-white/5 dark:from-gray-700/20 dark:via-gray-700/10 dark:to-gray-700/5 pointer-events-none" />

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6 relative min-h-0">
            <AnimatePresence mode="wait">
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
                className="flex items-start space-x-3 w-full"
              >
                <motion.div
                  className="w-8 h-8 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg"
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  <Bot className="w-4 h-4 text-white" />
                </motion.div>
                <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl px-4 py-3 shadow-lg border border-white/50 dark:border-gray-600/50 max-w-xs lg:max-w-md">
                  <TypingIndicator />
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <motion.div
            className="p-6 border-t border-white/30 dark:border-gray-600/30 bg-white/20 dark:bg-gray-800/20 backdrop-blur-sm relative"
            animate={{
              backgroundColor: isInputFocused
                ? isDark
                  ? "rgba(55, 65, 81, 0.4)"
                  : "rgba(255, 255, 255, 0.4)"
                : isDark
                ? "rgba(55, 65, 81, 0.2)"
                : "rgba(255, 255, 255, 0.2)",
            }}
            transition={{ duration: 0.3 }}
          >
            <div className="flex items-center space-x-4">
              <div className="flex-1">
                <AnimatedTextarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  onFocus={() => setIsInputFocused(true)}
                  onBlur={() => setIsInputFocused(false)}
                  placeholder="✨ Tell me about your dream room... What's your style, budget, and vibe?"
                  isLoading={isLoading}
                  disabled={isLoading}
                />
              </div>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading}
                className={`w-12 h-12 rounded-2xl flex items-center justify-center transition-all duration-300 shadow-lg self-end ${
                  inputValue.trim() && !isLoading
                    ? "bg-gradient-to-br from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-xl hover:shadow-2xl"
                    : "bg-gray-200/80 dark:bg-gray-600/80 text-gray-400 dark:text-gray-500 cursor-not-allowed backdrop-blur-sm"
                }`}
              >
                {isLoading ? (
                  <motion.div
                    className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
                    animate={{ rotate: 360 }}
                    transition={{
                      duration: 1,
                      repeat: Infinity,
                      ease: "linear",
                    }}
                  />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </motion.button>
            </div>

            <div className="mt-4 flex items-center justify-between">
              <div className="text-xs text-gray-500 dark:text-gray-400 font-medium">
                Press Enter to send, Shift+Enter for new line
              </div>
              <div className="flex space-x-2">
                <motion.button
                  onClick={handleLoadingButton}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  disabled={isLoading}
                  className={`text-xs px-3 py-1.5 rounded-xl transition-all duration-200 backdrop-blur-sm border font-medium shadow-sm ${
                    isLoading
                      ? "bg-gradient-to-r from-pink-500 to-orange-500 text-white border-pink-400/50 cursor-not-allowed"
                      : "bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white border-blue-400/50 hover:shadow-lg"
                  }`}
                >
                  {isLoading ? (
                    <div className="flex items-center space-x-1">
                      <motion.div
                        className="w-3 h-3 border border-white border-t-transparent rounded-full"
                        animate={{ rotate: 360 }}
                        transition={{
                          duration: 1,
                          repeat: Infinity,
                          ease: "linear",
                        }}
                      />
                      <span>Loading...</span>
                    </div>
                  ) : (
                    "Test Loading Effect"
                  )}
                </motion.button>
                <motion.button
                  onClick={handleTestProductResponse}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="text-xs bg-white/60 dark:bg-gray-700/60 hover:bg-white/80 dark:hover:bg-gray-700/80 text-gray-600 dark:text-gray-300 px-3 py-1.5 rounded-xl transition-all duration-200 backdrop-blur-sm border border-white/50 dark:border-gray-600/50 font-medium shadow-sm"
                >
                  Test Products
                </motion.button>
                <motion.button
                  onClick={handleTestCategoryNotFound}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="text-xs bg-white/60 dark:bg-gray-700/60 hover:bg-white/80 dark:hover:bg-gray-700/80 text-gray-600 dark:text-gray-300 px-3 py-1.5 rounded-xl transition-all duration-200 backdrop-blur-sm border border-white/50 dark:border-gray-600/50 font-medium shadow-sm"
                >
                  Test Category Not Found
                </motion.button>
                <motion.button
                  onClick={handleTestBudgetConstraint}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="text-xs bg-white/60 dark:bg-gray-700/60 hover:bg-white/80 dark:hover:bg-gray-700/80 text-gray-600 dark:text-gray-300 px-3 py-1.5 rounded-xl transition-all duration-200 backdrop-blur-sm border border-white/50 dark:border-gray-600/50 font-medium shadow-sm"
                >
                  Test Budget Constraint
                </motion.button>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
}

export default App;
