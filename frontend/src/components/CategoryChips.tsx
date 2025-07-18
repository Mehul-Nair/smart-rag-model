import React from "react";
import { motion } from "framer-motion";
import { Sparkles } from "lucide-react";

interface CategoryChipsProps {
  categories: string[];
  onCategoryClick: (category: string) => void;
}

const CategoryChips: React.FC<CategoryChipsProps> = ({
  categories,
  onCategoryClick,
}) => {
  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center space-x-2">
        <Sparkles className="w-4 h-4 text-primary-500" />
        <span className="text-sm font-medium text-gray-700">
          Available Categories
        </span>
      </div>

      {/* Category Chips */}
      <div className="flex flex-wrap gap-2">
        {categories.map((category, index) => (
          <motion.button
            key={category}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.3, delay: index * 0.1 }}
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => onCategoryClick(category)}
            className="px-4 py-2 bg-gradient-to-r from-primary-50 to-primary-100 border border-primary-200 rounded-full text-sm font-medium text-primary-700 hover:from-primary-100 hover:to-primary-200 hover:border-primary-300 hover:text-primary-800 transition-all duration-200 shadow-sm hover:shadow-md cursor-pointer"
          >
            {category
              .replace("-", " ")
              .replace(/\b\w/g, (l) => l.toUpperCase())}
          </motion.button>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="pt-2">
        <p className="text-xs text-gray-500 mb-2">Quick actions:</p>
        <div className="flex flex-wrap gap-2">
          {["under 10k", "under 5k", "best rated"].map((action, index) => (
            <motion.button
              key={action}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{
                duration: 0.3,
                delay: (categories.length + index) * 0.1,
              }}
              whileHover={{ scale: 1.05, y: -1 }}
              whileTap={{ scale: 0.95 }}
              onClick={() =>
                onCategoryClick(`show me ${categories[0]} ${action}`)
              }
              className="px-3 py-1.5 bg-gray-100 border border-gray-200 rounded-full text-xs font-medium text-gray-600 hover:bg-gray-200 hover:border-gray-300 hover:text-gray-700 transition-all duration-200 cursor-pointer"
            >
              {action}
            </motion.button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default CategoryChips;
