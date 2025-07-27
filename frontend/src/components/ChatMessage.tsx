import React from "react";
import { motion } from "framer-motion";
import { Bot, User } from "lucide-react";
import ProductResponse from "./ProductResponse";
import CategoryResponse from "./CategoryResponse";
import CategoryNotFoundResponse from "./CategoryNotFoundResponse";
import BudgetConstraintResponse from "./BudgetConstraintResponse";

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

interface ChatMessageProps {
  message: Message;
  onCategoryClick?: (category: string) => void;
}

const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  onCategoryClick,
}) => {
  const isUser = message.sender === "user";

  console.log("ChatMessage content:", message.content, typeof message.content); // Debug log
  console.log("Is user:", isUser);
  console.log(
    "Content type check:",
    typeof message.content === "object" && "type" in message.content
  );

  // Handle different response types
  if (
    !isUser &&
    typeof message.content === "object" &&
    "type" in message.content
  ) {
    const responseType = message.content.type;
    console.log("Response type in ChatMessage:", responseType);

    if (responseType === "product_suggestion") {
      console.log("Rendering ProductResponse component"); // Debug log
      return (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
          className="w-full"
        >
          <ProductResponse
            data={message.content as ProductResponseData}
            timestamp={message.timestamp}
            onCategoryClick={onCategoryClick}
            userQuery={message.userQuery}
          />
        </motion.div>
      );
    }

    if (responseType === "category_not_found") {
      console.log("Rendering CategoryNotFoundResponse component"); // Debug log
      return (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
          className="w-full"
        >
          <CategoryNotFoundResponse
            data={message.content as CategoryNotFoundData}
            timestamp={message.timestamp}
            onCategoryClick={onCategoryClick || (() => {})}
          />
        </motion.div>
      );
    }

    if (responseType === "budget_constraint") {
      console.log("Rendering BudgetConstraintResponse component"); // Debug log
      return (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
          className="w-full"
        >
          <BudgetConstraintResponse
            data={message.content as BudgetConstraintData}
            timestamp={message.timestamp}
            onCategoryClick={onCategoryClick}
            userQuery={message.userQuery}
          />
        </motion.div>
      );
    }

    // Handle clarification response
    if (responseType === "clarification" && "message" in message.content) {
      return (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
          className="w-full"
        >
          <div className="inline-block px-5 py-3 rounded-2xl shadow-lg max-w-xs lg:max-w-md backdrop-blur-sm border bg-white/80 dark:bg-gray-800/80 text-gray-800 dark:text-gray-100 border-white/50 dark:border-gray-600/50">
            <div className="text-sm leading-relaxed whitespace-pre-wrap font-medium">
              {message.content.message}
            </div>
          </div>
          <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            {message.timestamp.toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </div>
        </motion.div>
      );
    }
  }

  // Handle category response
  if (!isUser && Array.isArray(message.content) && onCategoryClick) {
    console.log("Rendering CategoryResponse component"); // Debug log
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.4, ease: "easeOut" }}
        className="w-full"
      >
        <CategoryResponse
          categories={message.content}
          timestamp={message.timestamp}
          onCategoryClick={onCategoryClick}
        />
      </motion.div>
    );
  }

  // Handle regular text message
  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -20, scale: 0.95 }}
      transition={{
        duration: 0.4,
        ease: [0.25, 0.46, 0.45, 0.94],
        type: "spring",
        stiffness: 300,
        damping: 30,
      }}
      className={`flex items-start space-x-3 w-full ${
        isUser ? "flex-row-reverse space-x-reverse" : ""
      }`}
    >
      {/* Avatar */}
      <motion.div
        className={`w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg ${
          isUser
            ? "bg-gradient-to-br from-blue-500 to-purple-600"
            : "bg-gradient-to-br from-slate-400 to-slate-600"
        }`}
        whileHover={{ scale: 1.1, rotate: 5 }}
        transition={{ type: "spring", stiffness: 400, damping: 10 }}
      >
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <Bot className="w-4 h-4 text-white" />
        )}
      </motion.div>

      {/* Message Bubble */}
      <div className={`flex-1 ${isUser ? "text-right" : "text-left"}`}>
        <motion.div
          className={`inline-block px-5 py-3 rounded-2xl shadow-lg max-w-xs lg:max-w-md backdrop-blur-sm border ${
            isUser
              ? "bg-gradient-to-br from-blue-500 to-purple-600 text-white border-white/20"
              : "bg-white/80 dark:bg-gray-800/80 text-gray-800 dark:text-gray-100 border-white/50 dark:border-gray-600/50"
          }`}
          whileHover={{ scale: 1.02, y: -2 }}
          transition={{ type: "spring", stiffness: 400, damping: 25 }}
        >
          <div className="text-sm leading-relaxed whitespace-pre-wrap font-medium">
            {typeof message.content === "string"
              ? message.content
              : "Product data received"}
          </div>
        </motion.div>

        {/* Timestamp */}
        <div
          className={`text-xs text-gray-500 dark:text-gray-400 mt-1 ${
            isUser ? "text-right" : "text-left"
          }`}
        >
          {message.timestamp.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </div>
      </div>
    </motion.div>
  );
};

export default ChatMessage;
