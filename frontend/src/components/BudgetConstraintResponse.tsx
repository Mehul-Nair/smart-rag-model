import React from "react";
import { motion } from "framer-motion";
import { Bot, DollarSign } from "lucide-react";
import BudgetSuggestion from "./BudgetSuggestion";

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
      className="flex items-start space-x-3"
    >
      {/* Avatar */}
      <div className="w-8 h-8 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full flex items-center justify-center flex-shrink-0">
        <Bot className="w-4 h-4 text-white" />
      </div>

      {/* Content */}
      <div className="flex-1">
        {/* Message */}
        <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-2xl px-4 py-3 border border-yellow-200 mb-4">
          <div className="flex items-center space-x-2 mb-2">
            <DollarSign className="w-4 h-4 text-yellow-600" />
            <span className="text-sm font-medium text-yellow-800">
              Budget Constraint
            </span>
          </div>
          <div className="text-sm leading-relaxed text-yellow-700">
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
            className="px-4 py-2 bg-primary-500 hover:bg-primary-600 text-white rounded-lg text-sm font-medium transition-colors duration-200"
          >
            Show All {data.category}
          </button>
        </div>

        {/* Timestamp */}
        <div className="text-xs text-gray-500 text-left">
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
