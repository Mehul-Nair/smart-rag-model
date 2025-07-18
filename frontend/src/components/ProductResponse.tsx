import React from "react";
import { motion } from "framer-motion";
import { Bot, Sparkles } from "lucide-react";
import ProductCard from "./ProductCard";
import BudgetSuggestion from "./BudgetSuggestion";

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

interface ProductResponseProps {
  data: ProductResponseData;
  timestamp: Date;
  onCategoryClick?: (category: string) => void;
  userQuery?: string;
}

const ProductResponse: React.FC<ProductResponseProps> = ({
  data,
  timestamp,
  onCategoryClick,
  userQuery,
}) => {
  // Extract budget from user query
  const extractBudget = (query: string): number | null => {
    const budgetMatch =
      query.match(/under\s*(\d+)/i) ||
      query.match(/(\d+)\s*k/i) ||
      query.match(/(\d+)/);
    if (budgetMatch) {
      let budget = parseInt(budgetMatch[1]);
      // If it's in thousands (k), multiply by 1000
      if (query.toLowerCase().includes("k")) {
        budget *= 1000;
      }
      return budget;
    }
    return null;
  };

  const originalBudget = userQuery ? extractBudget(userQuery) : null;
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
        {/* Summary */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl px-4 py-3 shadow-sm border border-gray-100 dark:border-gray-700 mb-4">
          <div className="flex items-center space-x-2 mb-2">
            <Sparkles className="w-4 h-4 text-primary-500" />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Product Suggestions
            </span>
          </div>
          <div className="text-sm leading-relaxed text-gray-800 dark:text-gray-200">
            {data.summary}
          </div>
        </div>

        {/* Products Grid or No Products Message */}
        {data.products.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-3">
            {data.products.map((product, index) => (
              <ProductCard key={index} product={product} index={index} />
            ))}
          </div>
        ) : (
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-2xl p-6 border border-yellow-200 dark:border-yellow-700/50 mb-3">
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-yellow-400 to-orange-400 rounded-full flex items-center justify-center mx-auto mb-4">
                <svg
                  className="w-8 h-8 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
                  />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">
                No Products Found
              </h3>
              <p className="text-gray-600 dark:text-gray-300 mb-4">
                {data.summary ||
                  "We couldn't find any products matching your criteria."}
              </p>
              <div className="space-y-4">
                <div className="flex flex-wrap gap-2 justify-center">
                  <button
                    onClick={() =>
                      onCategoryClick && onCategoryClick("show me all products")
                    }
                    className="px-4 py-2 bg-primary-500 hover:bg-primary-600 text-white rounded-lg text-sm font-medium transition-colors duration-200"
                  >
                    Show All Products
                  </button>
                </div>

                {/* Budget Suggestions */}
                {originalBudget && (
                  <BudgetSuggestion
                    originalBudget={originalBudget}
                    onBudgetSelect={(newBudget) => {
                      if (onCategoryClick) {
                        // Extract the category from the original query
                        const categoryMatch = userQuery?.match(
                          /(bedside\s*table|sofa|curtain|rug|fabric)/i
                        );
                        const category = categoryMatch
                          ? categoryMatch[1]
                          : "products";
                        onCategoryClick(`show me ${category} ${newBudget}`);
                      }
                    }}
                  />
                )}
              </div>
            </div>
          </div>
        )}

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

export default ProductResponse;
