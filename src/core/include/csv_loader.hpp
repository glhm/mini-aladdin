#pragma once
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include "option.hpp"

namespace mini_aladdin {

class CsvLoader {
public:
    static OptionBatch load(const std::string& filepath) {
        OptionBatch batch;
        std::ifstream file(filepath);
        
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open " << filepath << std::endl;
            return batch;
        }
        
        std::string line;
        // Skip header
        std::getline(file, line);
        
        // Parse data line by line
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            std::stringstream ss(line);
            std::string ticker;
            double S, K, r, sigma, T;
            int is_call_int;
            
            std::getline(ss, ticker, ',');
            ss >> S; ss.ignore(1, ',');
            ss >> K; ss.ignore(1, ',');
            ss >> r; ss.ignore(1, ',');
            ss >> sigma; ss.ignore(1, ',');
            ss >> T; ss.ignore(1, ',');
            ss >> is_call_int;
            
            batch.tickers.push_back(ticker);
            batch.S.push_back(S);
            batch.K.push_back(K);
            batch.r.push_back(r);
            batch.sigma.push_back(sigma);
            batch.T.push_back(T);
            batch.is_call.push_back(static_cast<int8_t>(is_call_int));
        }
        
        return batch;
    }
};

} // namespace mini_aladdin
