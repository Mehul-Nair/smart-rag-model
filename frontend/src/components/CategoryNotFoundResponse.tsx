import React from "react";
import { motion } from "framer-motion";
import { Bot, AlertCircle } from "lucide-react";
import CategoryChips from "./CategoryChips";

interface CategoryNotFoundData {
  type: string;
  requested_category: string;
  available_categories: string[];
  message: string;
}

interface CategoryNotFoundResponseProps {
  data: CategoryNotFoundData;
  timestamp: Date;
  onCategoryClick: (category: string) => void;
}

const CategoryNotFoundResponse: React.FC<CategoryNotFoundResponseProps> = ({
  data,
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
      <div className="w-8 h-8 bg-gradient-to-r from-orange-400 to-red-500 rounded-full flex items-center justify-center flex-shrink-0">
        <Bot className="w-4 h-4 text-white" />
      </div>

      {/* Content */}
      <div className="flex-1">
        {/* Error Message */}
        <div className="bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-2xl px-4 py-3 border border-orange-200 dark:border-orange-700/50 mb-4">
          <div className="flex items-center space-x-2 mb-2">
            <AlertCircle className="w-4 h-4 text-orange-600 dark:text-orange-400" />
            <span className="text-sm font-medium text-orange-800 dark:text-orange-300">
              Category Not Found
            </span>
          </div>
          <div className="text-sm leading-relaxed text-orange-700 dark:text-orange-300">
            We don't have <strong>{data.requested_category}</strong> in our
            catalog.
          </div>
        </div>

        {/* Available Categories */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-2xl p-4 border border-blue-100 dark:border-blue-800/50 mb-3">
          <div className="text-sm text-blue-800 dark:text-blue-300 mb-3">
            Here are the categories we do have:
          </div>
          <CategoryChips
            categories={data.available_categories}
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

export default CategoryNotFoundResponse;
