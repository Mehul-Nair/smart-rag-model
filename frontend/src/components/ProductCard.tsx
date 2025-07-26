import React from "react";
import { motion } from "framer-motion";
import { ExternalLink, Star, Image } from "lucide-react";

interface Product {
  name: string;
  price: string;
  url: string;
}

interface ProductCardProps {
  product: Product;
  index: number;
}

const ProductCard: React.FC<ProductCardProps> = ({ product, index }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{
        duration: 0.5,
        delay: index * 0.1,
        type: "spring",
        stiffness: 300,
        damping: 25,
      }}
      whileHover={{
        y: -8,
        scale: 1.02,
        transition: { type: "spring", stiffness: 400, damping: 25 },
      }}
      className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-sm rounded-2xl shadow-xl border border-white/50 dark:border-gray-600/50 overflow-hidden hover:shadow-2xl transition-all duration-300 group w-full"
    >
      {/* Product Image Placeholder */}
      <div className="h-48 bg-gradient-to-br from-gray-100 via-gray-50 to-gray-200 dark:from-gray-700 dark:via-gray-600 dark:to-gray-500 flex items-center justify-center relative overflow-hidden">
        {/* Subtle animated background */}
        <motion.div
          className="absolute inset-0 opacity-30"
          animate={{
            background: [
              "linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(147, 51, 234, 0.05) 100%)",
              "linear-gradient(135deg, rgba(147, 51, 234, 0.05) 0%, rgba(236, 72, 153, 0.05) 100%)",
              "linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(147, 51, 234, 0.05) 100%)",
            ],
          }}
          transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        />
        <div className="text-center z-10">
          <motion.div
            className="w-20 h-20 bg-gradient-to-br from-gray-300 to-gray-400 dark:from-gray-500 dark:to-gray-600 rounded-2xl flex items-center justify-center mx-auto mb-3 shadow-lg"
            whileHover={{ scale: 1.05 }}
            transition={{ type: "spring", stiffness: 400, damping: 10 }}
          >
            <Image className="w-10 h-10 text-gray-500 dark:text-gray-300" />
          </motion.div>
          <p className="text-sm text-gray-500 dark:text-gray-400 font-medium">
            Product Image
          </p>
        </div>
      </div>

      {/* Product Info */}
      <div className="p-5">
        <div className="flex items-start justify-between mb-3">
          <h3 className="font-bold text-gray-900 dark:text-gray-100 text-sm leading-tight flex-1 mr-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors duration-200">
            {product.name}
          </h3>
          <div className="flex items-center space-x-1 flex-shrink-0">
            <motion.div
              whileHover={{ scale: 1.2, rotate: 72 }}
              transition={{ type: "spring", stiffness: 400, damping: 10 }}
            >
              <Star className="w-3 h-3 text-yellow-400 fill-current" />
            </motion.div>
            <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">
              4.5
            </span>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="text-lg font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            ₹{product.price}
          </div>
          <motion.a
            href={product.url}
            target="_blank"
            rel="noopener noreferrer"
            whileHover={{ scale: 1.05, y: -2 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center space-x-1 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-200 shadow-lg hover:shadow-xl"
          >
            <ExternalLink className="w-3 h-3" />
            <span>View</span>
          </motion.a>
        </div>
      </div>
    </motion.div>
  );
};

export default ProductCard;
