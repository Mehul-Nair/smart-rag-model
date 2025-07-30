import React from "react";
import { motion } from "framer-motion";
import { ExternalLink, Star, Image, Heart } from "lucide-react";

interface Product {
  name: string;
  price: string;
  url: string;
  featuredImg?: string;
  discounted_price?: string;
  discount_percentage?: string;
}

interface ProductCardProps {
  product: Product;
  index: number;
}

const ProductCard: React.FC<ProductCardProps> = ({ product, index }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 50, scale: 0.8 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{
        duration: 0.6,
        delay: index * 0.15, // Staggered delay - each card appears 0.15s after the previous one
        type: "spring",
        stiffness: 200,
        damping: 20,
      }}
      whileHover={{
        y: -8,
        scale: 1.02,
        transition: { type: "spring", stiffness: 400, damping: 25 },
      }}
      className="bg-white rounded-lg shadow-sm border border-gray-200 group w-full relative max-w-sm"
    >
             {/* Heart Icon */}
       <div className="absolute top-3 right-3 z-10">
         <motion.button
           whileHover={{ scale: 1.1 }}
           whileTap={{ scale: 0.9 }}
           className="w-8 h-8 bg-white/80 backdrop-blur-sm rounded-full border border-gray-200 flex items-center justify-center hover:bg-white transition-colors duration-200"
         >
           <Heart className="w-4 h-4 text-gray-400 hover:text-red-500 transition-colors duration-200" />
         </motion.button>
       </div>

               {/* Discount Label */}
        {product.discount_percentage && (
          <div className="absolute top-3 left-3 z-10">
            <div className="bg-custom-purple text-white text-xs font-medium tracking-wider leading-3 px-1.5 py-1.5 rounded-sm">
              {product.discount_percentage} OFF
            </div>
          </div>
        )}

      {/* Product Image */}
      <div className="h-48 bg-gray-100 flex items-center justify-center relative overflow-hidden rounded-t-lg">
        {product.featuredImg ? (
          <img
            src={`https://www.beautifulhomes.asianpaints.com${product.featuredImg}`}
            alt={product.name}
            className="w-full h-full object-cover"
            onError={(e) => {
              // Fallback to placeholder if image fails to load
              const target = e.target as HTMLImageElement;
              target.style.display = 'none';
              target.nextElementSibling?.classList.remove('hidden');
            }}
          />
        ) : null}
        
        {/* Fallback Placeholder */}
        <div className={`text-center z-10 ${product.featuredImg ? 'hidden' : ''}`}>
          <div className="w-16 h-16 bg-gray-300 rounded-lg flex items-center justify-center mx-auto mb-3">
            <Image className="w-8 h-8 text-gray-500" />
          </div>
          <p className="text-sm text-gray-500 font-medium">
            Product Image
          </p>
        </div>
      </div>

      {/* Product Info */}
      <div className="p-3">
        {/* Product Name */}
        <h3 className="font-semibold text-gray-900 text-sm leading-tight mb-1 group-hover:text-blue-600 transition-colors duration-200">
          {product.name}
        </h3>
        
        {/* Product Description */}
        <p className="text-xs text-gray-500 mb-2 line-clamp-2">
          Bronx furniture embodies sophisticated design with premium quality materials...
        </p>

                 {/* Price Section */}
         <div className="flex items-center justify-between mb-2">
                       <div className="flex flex-col">
              <div className="flex items-center gap-2">
                <span className="text-base font-bold text-gray-900">
                  ₹{product.discounted_price || product.price}
                </span>
                {/* {product.discount_percentage && (
                  <span className="bg-custom-purple text-white text-xs font-medium tracking-wider leading-3 px-1.5 py-1.5 rounded-sm">
                    {product.discount_percentage}% OFF
                  </span>
                )} */}
              </div>
              <div className="flex items-center gap-1">
                <span className="text-xs text-gray-400 line-through">
                  MRP ₹{product.price}
                </span>
              </div>
            </div>
          
                     {/* Add to Cart Button */}
           <motion.a
             href={`https://www.beautifulhomes.asianpaints.com${product.url}`}
             target="_blank"
             rel="noopener noreferrer"
             whileHover={{ scale: 1.05 }}
             whileTap={{ scale: 0.95 }}
             className="w-10 h-10 bg-white border border-gray-300 rounded-full flex items-center justify-center hover:border-gray-400 transition-colors duration-200"
           >
             <span className="text-purple-600 text-xl font-bold">+</span>
           </motion.a>
        </div>
      </div>
    </motion.div>
  );
};

export default ProductCard;
