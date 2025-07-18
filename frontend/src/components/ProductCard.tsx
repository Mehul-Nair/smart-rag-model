import React from "react";
import { motion } from "framer-motion";
import { ExternalLink, ShoppingCart, Star } from "lucide-react";

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
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: index * 0.1 }}
      className="bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden hover:shadow-xl transition-all duration-300 hover:-translate-y-1"
    >
      {/* Product Image Placeholder */}
      <div className="h-48 bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-gradient-to-r from-primary-400 to-primary-600 rounded-full flex items-center justify-center mx-auto mb-2">
            <ShoppingCart className="w-8 h-8 text-white" />
          </div>
          <p className="text-sm text-gray-600 font-medium">Product Image</p>
        </div>
      </div>

      {/* Product Info */}
      <div className="p-4">
        <div className="flex items-start justify-between mb-2">
          <h3 className="font-semibold text-gray-900 text-sm leading-tight max-h-10 overflow-hidden">
            {product.name}
          </h3>
          <div className="flex items-center space-x-1 ml-2">
            <Star className="w-3 h-3 text-yellow-400 fill-current" />
            <span className="text-xs text-gray-500">4.5</span>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="text-lg font-bold text-primary-600">
            {product.price}
          </div>
          <motion.a
            href={product.url}
            target="_blank"
            rel="noopener noreferrer"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center space-x-1 bg-primary-500 hover:bg-primary-600 text-white px-3 py-1.5 rounded-lg text-sm font-medium transition-colors duration-200"
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
