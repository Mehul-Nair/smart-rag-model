import React from "react";
import { motion } from "framer-motion";
import { TrendingUp, DollarSign, IndianRupee } from "lucide-react";

interface BudgetSuggestionProps {
  originalBudget: number;
  onBudgetSelect: (newBudget: string) => void;
}

const BudgetSuggestion: React.FC<BudgetSuggestionProps> = ({
  originalBudget,
  onBudgetSelect,
}) => {
  // Calculate percentage increases
  const suggestions = [
    { percentage: 10, label: "10% more" },
    { percentage: 20, label: "20% more" },
    { percentage: 30, label: "30% more" },
  ];

  const calculateNewBudget = (percentage: number) => {
    return Math.round(originalBudget * (1 + percentage / 100));
  };

           return (
           <motion.div
         initial={{ opacity: 0, y: 10 }}
         animate={{ opacity: 1, y: 0 }}
         className="rounded-xl p-4 border border-blue-200 dark:border-blue-700/50 bg-gradient-to-br from-purple-50 via-pink-150 to-orange-100"
       >
       <div className="flex items-center space-x-2 mb-3">
         <TrendingUp className="w-4 h-4 text-black" />
         <span className="text-sm font-medium text-black">
           Budget Suggestions
         </span>
       </div>

       <p className="text-xs text-black mb-3">
         Try increasing your budget to find more options:
       </p>

      <div className="flex flex-wrap gap-2">
        {suggestions.map((suggestion, index) => {
          const newBudget = calculateNewBudget(suggestion.percentage);
          return (
            <motion.button
              key={suggestion.percentage}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              whileHover={{ scale: 1.05, y: -1 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => onBudgetSelect(`under ${newBudget}`)}
                             className="flex items-center space-x-2 px-3 py-2 bg-white dark:bg-gray-800 border border-blue-200 dark:border-blue-600 rounded-lg text-sm font-medium text-black hover:bg-blue-50 dark:hover:bg-gray-700 hover:border-blue-300 dark:hover:border-blue-500 transition-all duration-200 shadow-sm hover:shadow-md"
             >
               <IndianRupee className="w-3 h-3 text-black" />
               <span className="text-black">{suggestion.label}</span>
               <span className="text-xs text-black">
                 ({newBudget.toLocaleString()})
               </span>
            </motion.button>
          );
        })}
      </div>
    </motion.div>
  );
};

export default BudgetSuggestion;
