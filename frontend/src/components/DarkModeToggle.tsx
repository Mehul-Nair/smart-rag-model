import React from "react";
import { motion } from "framer-motion";
import { Sun, Moon } from "lucide-react";

interface DarkModeToggleProps {
  isDark: boolean;
  onToggle: () => void;
}

const DarkModeToggle: React.FC<DarkModeToggleProps> = ({
  isDark,
  onToggle,
}) => {
  return (
    <motion.button
      onClick={onToggle}
      className="relative w-12 h-6 bg-gray-200 dark:bg-gray-700 rounded-full p-1 transition-colors duration-300 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800"
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
    >
      <motion.div
        className="w-4 h-4 bg-white dark:bg-white rounded-full shadow-lg dark:shadow-xl flex items-center justify-center relative z-10"
        animate={{
          x: isDark ? 24 : 0,
        }}
        transition={{
          type: "spring",
          stiffness: 500,
          damping: 30,
        }}
      >
        <motion.div
          initial={false}
          animate={{
            rotate: isDark ? 180 : 0,
            scale: isDark ? 1 : 0.8,
          }}
          transition={{
            duration: 0.3,
            ease: "easeInOut",
          }}
          className="relative z-20"
        >
          {isDark ? (
            <Moon className="w-3 h-3 text-gray-800" />
          ) : (
            <Sun className="w-3 h-3 text-yellow-500" />
          )}
        </motion.div>
      </motion.div>

      {/* Background gradient animation */}
      <motion.div
        className="absolute inset-0 rounded-full opacity-0 dark:opacity-100 z-0"
        style={{
          background: "linear-gradient(135deg, #64748b 0%, #94a3b8 100%)",
        }}
        initial={false}
        animate={{
          opacity: isDark ? 1 : 0,
        }}
        transition={{
          duration: 0.3,
        }}
      />
    </motion.button>
  );
};

export default DarkModeToggle;
