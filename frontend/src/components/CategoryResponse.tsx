import React from "react";
import { motion } from "framer-motion";
import { Bot } from "lucide-react";
import CategoryChips from "./CategoryChips";

interface CategoryResponseProps {
  categories: string[];
  timestamp: Date;
  onCategoryClick: (category: string) => void;
}

const CategoryResponse: React.FC<CategoryResponseProps> = ({
  categories,
  timestamp,
  onCategoryClick,
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="flex items-start space-x-3"
    >
      {/* Avatar */}
      <div className="w-8 h-8 bg-gradient-to-r from-primary-400 to-primary-500 rounded-full flex items-center justify-center flex-shrink-0">
        <Bot className="w-4 h-4 text-white" />
      </div>

      {/* Content */}
      <div className="flex-1">
        {/* Message */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl px-4 py-3 shadow-sm border border-gray-100 dark:border-gray-700 mb-4">
          <div className="text-sm leading-relaxed text-gray-800 dark:text-gray-200">
            I have the following product categories available:
          </div>
        </div>

        {/* Category Chips */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-2xl p-4 border border-blue-100 dark:border-blue-800/50 mb-3">
          <CategoryChips
            categories={categories}
            onCategoryClick={onCategoryClick}
          />
        </div>

        {/* Timestamp */}
        <div className="text-xs text-gray-500 dark:text-gray-400 text-left">
          {timestamp.toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </div>
      </div>
    </motion.div>
  );
};

export default CategoryResponse;
