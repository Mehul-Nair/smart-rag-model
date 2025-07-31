import React from "react";
import { motion } from "framer-motion";
import { Bot, IndianRupee } from "lucide-react";
import BudgetSuggestion from "./BudgetSuggestion";
import logoImage from "../assets/images/logos/asian-paint-ap-logo.png";

interface BudgetConstraintData {
  type: string;
  category: string;
  requested_budget: string;
  message: string;
}

interface BudgetConstraintResponseProps {
  data: BudgetConstraintData;
  timestamp: Date;
  onCategoryClick?: (category: string) => void;
  userQuery?: string;
}

const BudgetConstraintResponse: React.FC<BudgetConstraintResponseProps> = ({
  data,
  timestamp,
  onCategoryClick,
  userQuery,
}) => {
  // Extract budget from requested_budget
  const extractBudget = (budgetStr: string): number | null => {
    const budgetMatch = budgetStr.match(/(\d+)/);
    if (budgetMatch) {
      let budget = parseInt(budgetMatch[1]);
      // If it's in thousands (k), multiply by 1000
      if (budgetStr.toLowerCase().includes("k")) {
        budget *= 1000;
      }
      return budget;
    }
    return null;
  };

  const originalBudget = extractBudget(data.requested_budget);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="flex items-start space-x-3 flex-col gap-5"
    >
      {/* Avatar */}
      <div className="w-10 h-10 rounded-full p-2 flex items-center justify-center flex-shrink-0 shadow-lg">
      <img
          src={logoImage}
          alt="Beautiful Homes Logo AP"
          width="60"
          height="75"
          className="sm:w-20 sm:h-25"
        />
      </div>

      {/* Content */}
      <div className="w-full">
        {/* Message */}
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-2xl px-4 py-3 border border-yellow-200 dark:border-yellow-700/50 mb-4">
          <div className="flex items-center space-x-2 mb-2">
            <IndianRupee className="w-4 h-4 text-yellow-600 dark:text-yellow-400" />
            <span className="text-sm font-medium text-yellow-800 dark:text-yellow-300">
              Budget Constraint
            </span>
          </div>
          <div className="text-sm leading-relaxed text-yellow-700 dark:text-yellow-300">
            We found <strong>{data.category}</strong> but none under your budget
            of <strong>{data.requested_budget}</strong>.
          </div>
        </div>

        {/* Budget Suggestions */}
        {originalBudget && (
          <div className="mb-3">
            <BudgetSuggestion
              originalBudget={originalBudget}
              onBudgetSelect={(newBudget) => {
                if (onCategoryClick) {
                  onCategoryClick(`show me ${data.category} ${newBudget}`);
                }
              }}
            />
          </div>
        )}

        {/* Alternative Actions */}
        <div className="flex flex-wrap gap-2 justify-center mb-3">
          <button
            onClick={() =>
              onCategoryClick && onCategoryClick(`show me ${data.category}`)
            }
            className="px-4 py-2 bg-custom-purple hover:bg-custom-purple-dark text-white rounded-lg text-sm font-medium transition-colors duration-200"
          >
            Show All {data.category}
          </button>
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

export default BudgetConstraintResponse;
