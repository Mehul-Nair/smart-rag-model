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

  // Handle different response types
  if (
    !isUser &&
    typeof message.content === "object" &&
    "type" in message.content
  ) {
    const responseType = message.content.type;

    if (responseType === "product_suggestion") {
      console.log("Rendering ProductResponse component"); // Debug log
      return (
        <ProductResponse
          data={message.content as ProductResponseData}
          timestamp={message.timestamp}
          onCategoryClick={onCategoryClick}
          userQuery={message.userQuery}
        />
      );
    }

    if (responseType === "category_not_found") {
      console.log("Rendering CategoryNotFoundResponse component"); // Debug log
      return (
        <CategoryNotFoundResponse
          data={message.content as CategoryNotFoundData}
          timestamp={message.timestamp}
          onCategoryClick={onCategoryClick || (() => {})}
        />
      );
    }

    if (responseType === "budget_constraint") {
      console.log("Rendering BudgetConstraintResponse component"); // Debug log
      return (
        <BudgetConstraintResponse
          data={message.content as BudgetConstraintData}
          timestamp={message.timestamp}
          onCategoryClick={onCategoryClick}
          userQuery={message.userQuery}
        />
      );
    }
  }

  // Handle category response
  if (!isUser && Array.isArray(message.content) && onCategoryClick) {
    console.log("Rendering CategoryResponse component"); // Debug log
    return (
      <CategoryResponse
        categories={message.content}
        timestamp={message.timestamp}
        onCategoryClick={onCategoryClick}
      />
    );
  }

  // Handle regular text message
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className={`flex items-start space-x-3 ${
        isUser ? "flex-row-reverse space-x-reverse" : ""
      }`}
    >
      {/* Avatar */}
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser
            ? "bg-gradient-to-r from-primary-500 to-primary-600"
            : "bg-gradient-to-r from-gray-400 to-gray-500"
        }`}
      >
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <Bot className="w-4 h-4 text-white" />
        )}
      </div>

      {/* Message Bubble */}
      <div className={`flex-1 ${isUser ? "text-right" : "text-left"}`}>
        <div
          className={`inline-block px-4 py-3 rounded-2xl shadow-sm max-w-xs lg:max-w-md ${
            isUser
              ? "bg-primary-500 text-white"
              : "bg-white text-gray-800 border border-gray-100"
          }`}
        >
          <div className="text-sm leading-relaxed whitespace-pre-wrap">
            {typeof message.content === "string"
              ? message.content
              : "Product data received"}
          </div>
        </div>

        {/* Timestamp */}
        <div
          className={`text-xs text-gray-500 mt-1 ${
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
