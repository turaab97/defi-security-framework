# 🔵 Google BigQuery Integration Guide

## Overview

Your DeFi Security Framework now supports **Google BigQuery public blockchain datasets** with data from 14+ blockchains!

---

## 📊 Available BigQuery Datasets

### **Ethereum (Primary)**
```
bigquery-public-data.crypto_ethereum.transactions
bigquery-public-data.crypto_ethereum.blocks
bigquery-public-data.crypto_ethereum.token_transfers
bigquery-public-data.crypto_ethereum.contracts
bigquery-public-data.crypto_ethereum.logs
bigquery-public-data.crypto_ethereum.traces
```

### **Other Chains (if available)**
- `crypto_polygon` - Polygon
- `crypto_arbitrum` - Arbitrum
- `crypto_optimism` - Optimism
- `crypto_base` - Base
- `crypto_bsc` - BNB Chain
- `crypto_avalanche` - Avalanche
- And more!

---

## 🚀 Quick Start (No Auth Required for Public Data)

### **Option 1: Run Without Authentication (Limited)**
```bash
cd ~/Downloads/defi-security-framework
python3 defi_bigquery.py
```

The system will fall back to Web3 if BigQuery auth isn't set up.

---

## 🔑 Full Setup with BigQuery Authentication

### **Step 1: Create Google Cloud Project (Free)**

1. **Go to:**
   - https://console.cloud.google.com/

2. **Create new project:**
   - Click "Select a project" → "New Project"
   - Name: `defi-security-framework`
   - Click "Create"

3. **Enable BigQuery API:**
   - Go to: https://console.cloud.google.com/apis/library/bigquery.googleapis.com
   - Click "Enable"

### **Step 2: Set Up Authentication**

#### **Method A: Application Default Credentials (Easiest)**
```bash
# Install Google Cloud SDK (if not installed)
# Mac:
brew install google-cloud-sdk

# Linux:
curl https://sdk.cloud.google.com | bash

# Then authenticate:
gcloud auth application-default login
```

#### **Method B: Service Account (For Production)**

1. **Create service account:**
   - Go to: https://console.cloud.google.com/iam-admin/serviceaccounts
   - Click "Create Service Account"
   - Name: `defi-bigquery-reader`
   - Click "Create and Continue"
   - Role: "BigQuery Data Viewer"
   - Click "Done"

2. **Create key:**
   - Click on the service account
   - Keys → Add Key → Create New Key
   - Type: JSON
   - Download the key file

3. **Set environment variable:**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key.json"
   
   # Add to ~/.zshrc or ~/.bashrc for persistence:
   echo 'export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/key.json"' >> ~/.zshrc
   ```

### **Step 3: Install Required Packages**
```bash
pip install google-cloud-bigquery db-dtypes
```

### **Step 4: Run with BigQuery**
```bash
cd ~/Downloads/defi-security-framework
python3 defi_bigquery.py
```

You should see:
```
✓ BigQuery client available
✓ BigQuery client initialized
🔍 Querying BigQuery for ethereum transactions...
✓ Fetched 20 transactions from BigQuery
```

---

## 🎯 What Data is Available

### **1. Transactions**
```sql
SELECT 
    hash,
    from_address,
    to_address,
    value,
    gas,
    gas_price,
    block_number,
    block_timestamp
FROM `bigquery-public-data.crypto_ethereum.transactions`
LIMIT 10
```

### **2. Token Transfers**
```sql
SELECT 
    transaction_hash,
    from_address,
    to_address,
    value,
    token_address
FROM `bigquery-public-data.crypto_ethereum.token_transfers`
LIMIT 10
```

### **3. Blocks**
```sql
SELECT 
    number,
    timestamp,
    transaction_count,
    gas_used,
    gas_limit
FROM `bigquery-public-data.crypto_ethereum.blocks`
LIMIT 10
```

### **4. Contracts**
```sql
SELECT 
    address,
    bytecode,
    function_sighashes,
    is_erc20,
    is_erc721
FROM `bigquery-public-data.crypto_ethereum.contracts`
LIMIT 10
```

### **5. Logs**
```sql
SELECT 
    transaction_hash,
    log_index,
    address,
    data,
    topics
FROM `bigquery-public-data.crypto_ethereum.logs`
LIMIT 10
```

### **6. Traces**
```sql
SELECT 
    transaction_hash,
    from_address,
    to_address,
    value,
    call_type
FROM `bigquery-public-data.crypto_ethereum.traces`
LIMIT 10
```

---

## 📊 Supported Blockchains (14+)

| Blockchain | BigQuery | Web3 RPC | Native Token |
|------------|----------|----------|--------------|
| **Ethereum** | ✅ Yes | ✅ Yes | ETH |
| **Polygon** | ✅ Yes | ✅ Yes | MATIC |
| **Arbitrum** | ✅ Yes | ✅ Yes | ETH |
| **Optimism** | ✅ Yes | ✅ Yes | ETH |
| **Base** | ✅ Yes | ✅ Yes | ETH |
| **BNB Chain** | ✅ Yes | ✅ Yes | BNB |
| **zkSync Era** | 🔄 Partial | ✅ Yes | ETH |
| **Avalanche** | ✅ Yes | ✅ Yes | AVAX |
| **Fantom** | 🔄 Partial | ✅ Yes | FTM |
| **Cronos** | ❌ No | ✅ Yes | CRO |
| **Gnosis** | ❌ No | ✅ Yes | xDAI |
| **Celo** | ❌ No | ✅ Yes | CELO |
| **Moonbeam** | ❌ No | ✅ Yes | GLMR |
| **Aurora** | ❌ No | ✅ Yes | ETH |

---

## 🔄 Data Source Priority

The system uses a smart fallback system:

```
1. 🔵 BigQuery (if authenticated)
   ↓ (if not available)
2. 🔴 Web3 RPC (if connected)
   ↓ (if not available)
3. 🟡 Synthetic Data (always works)
```

This ensures the system always has data!

---

## 💰 BigQuery Costs

### **Free Tier:**
- **1 TB/month** of query processing free
- Public datasets are free to query
- Our queries use ~0.001 TB per 1000 transactions

### **Typical Usage:**
```
Transactions per day: 10,000
Data processed: ~0.01 TB/day
Monthly cost: $0 (within free tier!)
```

### **Monitor Usage:**
https://console.cloud.google.com/bigquery

---

## 🎯 How It Works

### **System Flow:**

```
┌─────────────────────────────────────────┐
│   DeFi Security Framework               │
├─────────────────────────────────────────┤
│                                         │
│  1. Try BigQuery                        │
│     └─> Query public datasets           │
│     └─> Get last hour transactions      │
│     └─> Cache results                   │
│                                         │
│  2. Fallback to Web3                    │
│     └─> Connect to RPC                  │
│     └─> Fetch latest blocks             │
│     └─> Extract transactions            │
│                                         │
│  3. Fallback to Synthetic               │
│     └─> Generate realistic data         │
│     └─> Maintain demo functionality     │
│                                         │
│  4. ML Analysis                         │
│     └─> Extract features                │
│     └─> Predict risk                    │
│     └─> Generate alerts                 │
│                                         │
│  5. Dashboard                           │
│     └─> Show data source (🔵🔴🟡)        │
│     └─> Real-time updates               │
│     └─> Interactive charts              │
└─────────────────────────────────────────┘
```

---

## 🚨 Troubleshooting

### **Error: "Could not initialize BigQuery"**

**Solution:**
```bash
# Check if gcloud is installed
gcloud --version

# Authenticate
gcloud auth application-default login

# Or set service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
```

### **Error: "Permission denied"**

**Solution:**
- Make sure BigQuery API is enabled
- Check service account has "BigQuery Data Viewer" role
- Verify credentials file path is correct

### **No transactions showing**

**Solution:**
- BigQuery queries recent data (last hour)
- For older data, system falls back to Web3/Synthetic
- This is normal behavior!

### **"Quota exceeded"**

**Solution:**
- You've used > 1TB this month (unlikely)
- Wait for next month or enable billing
- System will auto-fallback to Web3

---

## 🎨 Dashboard Indicators

### **Data Source Badges:**

- **🔵 BigQuery** - Data from Google BigQuery
- **🔴 Web3** - Data from blockchain RPCs
- **🟡 Synthetic** - Generated test data

### **Status Badge:**

- **🔵 BIGQUERY + WEB3** - Both sources active
- **🔴 WEB3 ONLY** - Only RPC connections
- **🟡 DEMO MODE** - Synthetic data only

---

## 📝 Example Queries

### **Get High-Value Transactions:**
```sql
SELECT 
    hash,
    value,
    block_timestamp
FROM `bigquery-public-data.crypto_ethereum.transactions`
WHERE value > 1000000000000000000  -- > 1 ETH
ORDER BY block_timestamp DESC
LIMIT 100
```

### **Find Contract Creations:**
```sql
SELECT 
    hash,
    from_address,
    block_timestamp
FROM `bigquery-public-data.crypto_ethereum.transactions`
WHERE to_address IS NULL
ORDER BY block_timestamp DESC
LIMIT 100
```

### **Analyze Gas Prices:**
```sql
SELECT 
    AVG(gas_price / 1e9) as avg_gwei,
    MAX(gas_price / 1e9) as max_gwei,
    DATE(block_timestamp) as date
FROM `bigquery-public-data.crypto_ethereum.transactions`
WHERE block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
GROUP BY date
ORDER BY date DESC
```

---

## 🎯 Next Steps

1. **Set up authentication** (optional but recommended)
2. **Run the BigQuery version:**
   ```bash
   cd ~/Downloads/defi-security-framework
   python3 defi_bigquery.py
   ```
3. **Open dashboard:** http://localhost:8080
4. **Look for 🔵 indicators** showing BigQuery data!

---

## 📚 Resources

- **BigQuery Docs:** https://cloud.google.com/bigquery/docs
- **Public Datasets:** https://console.cloud.google.com/marketplace/product/ethereum/crypto-ethereum-blockchain
- **Query Syntax:** https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax
- **Python Client:** https://cloud.google.com/python/docs/reference/bigquery/latest

---

## ✨ Benefits of BigQuery

✅ **Historical Data** - Access years of blockchain data
✅ **Fast Queries** - Analyze millions of transactions in seconds
✅ **Free Tier** - 1TB/month free
✅ **Reliable** - Google's infrastructure
✅ **SQL Interface** - Easy to query
✅ **Multiple Chains** - One interface for all

---

**Ready to explore billions of real blockchain transactions!** 🚀

