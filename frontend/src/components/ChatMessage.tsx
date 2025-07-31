import React, { useState } from "react";
import { motion } from "framer-motion";
import { Bot, User, Edit, Copy, Check } from "lucide-react";
import ProductResponse from "./ProductResponse";
import ProductDetailResponse from "./ProductDetailResponse";
import CategoryResponse from "./CategoryResponse";
import CategoryNotFoundResponse from "./CategoryNotFoundResponse";
import BudgetConstraintResponse from "./BudgetConstraintResponse";
import WarrantyResponse from "./WarrantyResponse";
import logoImage from "../assets/images/logos/asian-paint-ap-logo.png";

interface Product {
  name: string;
  price: string;
  url: string;
  featuredImg?: string;
  discounted_price?: string;
  discount_percentage?: string;
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

interface ProductDetailData {
  type: string;
  product_name: string;
  details: Record<string, string>;
  message: string;
}

interface TextResponseData {
  type: string;
  message: string;
}

interface Message {
  id: string;
  content:
    | string
    | ProductResponseData
    | ProductDetailData
    | CategoryNotFoundData
    | BudgetConstraintData
    | CategoryListData
    | TextResponseData
    | string[];
  sender: "user" | "ai";
  timestamp: Date;
  userQuery?: string;
}

interface ChatMessageProps {
  message: Message;
  onCategoryClick?: (category: string) => void;
  onEditMessage?: (messageId: string, newContent: string) => void;
}

const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  onCategoryClick,
  onEditMessage,
}) => {
  const isUser = message.sender === "user";
  const [isHovered, setIsHovered] = useState(false);
  const [copied, setCopied] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState(
    typeof message.content === "string" ? message.content : ""
  );

  const handleCopy = async () => {
    const contentToCopy = typeof message.content === "string" 
      ? message.content 
      : JSON.stringify(message.content);
    
    try {
      await navigator.clipboard.writeText(contentToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy: ', err);
    }
  };

  const handleEdit = () => {
    setIsEditing(true);
  };

  const handleSaveEdit = () => {
    if (onEditMessage && editContent.trim()) {
      onEditMessage(message.id, editContent.trim());
    }
    setIsEditing(false);
  };

  const handleCancelEdit = () => {
    setEditContent(typeof message.content === "string" ? message.content : "");
    setIsEditing(false);
  };

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

    if (responseType === "product_detail") {
      console.log("Rendering ProductDetailResponse component"); // Debug log
      return (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
          className="w-full"
        >
          <ProductDetailResponse
            data={message.content as ProductDetailData}
            timestamp={message.timestamp}
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

    // Handle text response (like warranty info)
    if (responseType === "text" && "message" in message.content) {
      // Check if this is warranty-related content
      const messageText = message.content.message.toLowerCase();
      const warrantyKeywords = [
        "warranty",
        "warranties",
        "guarantee",
        "guaranteed",
        "coverage",
        "defects",
        "manufacturing",
        "craftsmanship",
        "six-month",
        "one-year",
        "two-year",
        "period",
      ];

      const isWarrantyRelated = warrantyKeywords.some((keyword) =>
        messageText.includes(keyword)
      );

      if (isWarrantyRelated) {
        console.log("Rendering WarrantyResponse component"); // Debug log
        return (
          <WarrantyResponse
            data={message.content as TextResponseData}
            timestamp={message.timestamp}
          />
        );
      }

      // Regular text response
      return (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
          className="w-full"
        >
          <div className="flex items-start space-x-3 chat-message-container">
            {/* Avatar */}
            <div className="w-8 h-8 bg-gradient-to-r from-blue-400 to-blue-500 rounded-full flex items-center justify-center flex-shrink-0">
              <Bot className="w-4 h-4 text-white" />
            </div>

            {/* Content */}
            <div className="flex-1 flex-stable">
              <div className="inline-block px-5 py-3 rounded-2xl shadow-lg max-w-xs lg:max-w-md backdrop-blur-sm border bg-white/80 dark:bg-gray-800/80 text-gray-800 dark:text-gray-100 border-white/50 dark:border-gray-600/50">
                <div className="text-sm leading-relaxed whitespace-pre-wrap font-medium">
                  {message.content.message}
                </div>
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400 mt-1 ml-1">
                {message.timestamp.toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </div>
            </div>
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
    <div
      className={`flex items-start space-x-3 w-full ${
        isUser ? "flex-row-reverse space-x-reverse" : ""
      }`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Avatar */}
      <motion.div
        className={`w-10 h-10 rounded-full p-2 flex items-center justify-center flex-shrink-0 shadow-lg ${
          isUser
            ? "bg-custom-purple"
            : ""
        }`}
        whileHover={{ scale: 1.1, rotate: 5 }}
        transition={{ type: "spring", stiffness: 400, damping: 10 }}
      >
        {isUser ? (
          <User className="w-4 h-4 text-white" />
        ) : (
          <img
          src={logoImage}
          alt="Beautiful Homes Logo AP"
          width="60"
          height="75"
          className="sm:w-20 sm:h-25"
        />
          // <Bot className="w-4 h-4 text-white" />
        )}
      </motion.div>

   

      {/* Message Bubble */}
      <div className={`${isUser ? "text-right" : "text-left"}`}>
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
          className={`inline-block px-5 py-3 rounded-2xl shadow-lg max-w-xs lg:max-w-md backdrop-blur-sm border ${
            isUser
              ? "bg-custom-purple text-white border-white/20"
              : "bg-white/80 dark:bg-gray-800/80 text-gray-800 dark:text-gray-100 border-white/50 dark:border-gray-600/50"
          }`}
          whileHover={{ scale: 1.02, y: -2 }}
        >
          {isEditing ? (
            <div className="space-y-2">
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="w-full p-2 text-sm bg-white/90 dark:bg-gray-700/90 rounded-lg border border-gray-300 dark:border-gray-600 text-gray-900 dark:text-gray-100 resize-none"
                rows={3}
                autoFocus
              />
              <div className="flex space-x-2">
                <button
                  onClick={handleSaveEdit}
                  className="px-3 py-1 text-xs bg-green-500 hover:bg-green-600 text-white rounded-lg transition-colors"
                >
                  Save
                </button>
                <button
                  onClick={handleCancelEdit}
                  className="px-3 py-1 text-xs bg-gray-500 hover:bg-gray-600 text-white rounded-lg transition-colors"
                >
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className="text-sm leading-relaxed whitespace-pre-wrap font-medium">
              {typeof message.content === "string"
                ? message.content
                : typeof message.content === "object" &&
                  "message" in message.content
                ? message.content.message
                : "Product data received"}
            </div>
          )}
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
                   {/* Edit and Copy Actions - Only for user messages */}
          {isUser && isHovered && !isEditing && (
         <motion.div
           initial={{ opacity: 0, y: 10 }}
           animate={{ opacity: 1, y: 0 }}
           exit={{ opacity: 0, y: 10 }}
           transition={{ duration: 0.2 }}
           className="flex space-x-2 flex-shrink-0"
         >
          {/* <button
            onClick={handleEdit}
            className="p-1.5 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-all duration-200"
            title="Edit message"
          >
            <Edit className="w-3.5 h-3.5" />
          </button> */}
          <button
            onClick={handleCopy}
            className="p-1.5 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-all duration-200"
            title="Copy message"
          >
            {copied ? (
              <Check className="w-3.5 h-3.5 text-green-500" />
            ) : (
              <Copy className="w-3.5 h-3.5" />
            )}
          </button>
        </motion.div>
      )}
    </div>
  );
};

export default ChatMessage;
