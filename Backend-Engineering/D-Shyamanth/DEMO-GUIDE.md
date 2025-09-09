# 🎯 Live Newsletter System Demo Guide

Complete step-by-step guide to demonstrate your newsletter system with n8n integration.

## 🚀 Part 1: Setup & Prerequisites

### 1. Start Your Express.js API
```bash
cd /Users/karthikvangapandu/Desktop/newsletter
npm start
```
✅ **Expected:** Server running at `http://localhost:3000`

### 2. Install n8n (if not already installed)
```bash
npm install n8n -g
```

### 3. Start n8n
```bash
n8n start
```
✅ **Expected:** n8n running at `http://localhost:5678`

## 📋 Part 2: Prepare Demo Data

### 1. Add Some Subscribers
```bash
# Subscriber 1
curl -X POST http://localhost:3000/subscribe \
  -H "Content-Type: application/json" \
  -d '{"email": "student1@university.edu"}'

# Subscriber 2  
curl -X POST http://localhost:3000/subscribe \
  -H "Content-Type: application/json" \
  -d '{"email": "teacher@university.edu"}'

# Subscriber 3
curl -X POST http://localhost:3000/subscribe \
  -H "Content-Type: application/json" \
  -d '{"email": "demo@example.com"}'
```

### 2. Create a Newsletter
```bash
curl -X POST http://localhost:3000/newsletters \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Welcome to Our Demo Newsletter!", 
    "content": "This is a demonstration of our automated newsletter system with n8n integration. Thanks for subscribing!"
  }'
```

### 3. Verify Setup
```bash
# Check subscribers
curl http://localhost:3000/subscribers

# Check newsletters
curl http://localhost:3000/newsletters

# Check initial logs (should be empty)
curl http://localhost:3000/logs
```

## 🔧 Part 3: Import n8n Workflow

### Method 1: Import JSON File
1. Open n8n at `http://localhost:5678`
2. Click **"New Workflow"**
3. Click **"Import from file"** or **"Import from URL"**
4. Select `n8n-simple-workflow.json` from your project folder
5. Click **"Save"** to save the workflow

### Method 2: Manual Setup (Alternative)
1. **Create new workflow** in n8n
2. **Add Webhook node:**
   - Method: POST
   - Path: `demo-newsletter`
   - Response Mode: "Respond to Webhook"

3. **Add HTTP Request node (Get Subscribers):**
   - URL: `http://localhost:3000/subscribers`
   - Method: GET

4. **Add HTTP Request node (Log Email Sent):**
   - URL: `http://localhost:3000/webhook/email-status`
   - Method: POST
   - Body: JSON
   ```json
   {
     "email": "{{ $json.body.testEmail || 'demo@example.com' }}",
     "status": "sent",
     "newsletterId": "{{ Date.now() }}",
     "timestamp": "{{ new Date().toISOString() }}"
   }
   ```

5. **Add Wait node:** 2 seconds

6. **Add HTTP Request node (Log Email Delivered):**
   - URL: `http://localhost:3000/webhook/email-status`
   - Method: POST
   - Body: JSON
   ```json
   {
     "email": "{{ $json.body.testEmail || 'demo@example.com' }}",
     "status": "delivered",
     "newsletterId": "{{ Date.now() }}",
     "timestamp": "{{ new Date().toISOString() }}"
   }
   ```

7. **Add Wait node:** 3 seconds

8. **Add HTTP Request node (Log Email Opened):**
   - URL: `http://localhost:3000/webhook/email-status`
   - Method: POST
   - Body: JSON
   ```json
   {
     "email": "{{ $json.body.testEmail || 'demo@example.com' }}",
     "status": "opened",
     "newsletterId": "{{ Date.now() }}",
     "timestamp": "{{ new Date().toISOString() }}"
   }
   ```

9. **Add Respond to Webhook node:**
   ```json
   {
     "success": true,
     "message": "Newsletter workflow completed successfully!",
     "demo_flow": {
       "subscribers_count": "{{ $node['Get Subscribers'].json.count || 0 }}",
       "email_tested": "{{ $json.body.testEmail || 'demo@example.com' }}",
       "statuses_logged": ["sent", "delivered", "opened"],
       "timestamp": "{{ new Date().toISOString() }}"
     },
     "next_steps": {
       "check_logs": "GET http://localhost:3000/logs",
       "view_subscribers": "GET http://localhost:3000/subscribers"
     }
   }
   ```

## 🎬 Part 4: Live Demonstration Script

### Demo Flow Overview
1. **Show current state** of the system
2. **Trigger n8n workflow** 
3. **Watch real-time logs** being created
4. **Verify results** in your API

### Step-by-Step Demo

#### 1. Show Current System State
```bash
echo "=== CURRENT SUBSCRIBERS ==="
curl http://localhost:3000/subscribers | jq

echo -e "\n=== CURRENT NEWSLETTERS ==="
curl http://localhost:3000/newsletters | jq

echo -e "\n=== CURRENT LOGS (should be empty) ==="
curl http://localhost:3000/logs | jq
```

#### 2. Get n8n Webhook URL
- In n8n, click on the **Webhook node**
- Copy the **Production URL** (something like: `http://localhost:5678/webhook/demo-newsletter`)

#### 3. Trigger the n8n Workflow
```bash
# Basic trigger
curl -X POST http://localhost:5678/webhook/demo-newsletter \
  -H "Content-Type: application/json" \
  -d '{}'

# OR with custom email
curl -X POST http://localhost:5678/webhook/demo-newsletter \
  -H "Content-Type: application/json" \
  -d '{"testEmail": "teacher@university.edu"}'
```

#### 4. Watch the Magic Happen! ✨
**In n8n interface:**
- You'll see the workflow executing in real-time
- Each node will light up as it processes
- You can click on nodes to see the data flowing through

**Expected Flow:**
1. ✅ Webhook triggers
2. ✅ Gets subscriber list from API
3. ✅ Logs "sent" status
4. ⏳ Waits 2 seconds
5. ✅ Logs "delivered" status  
6. ⏳ Waits 3 seconds
7. ✅ Logs "opened" status
8. ✅ Returns success response

#### 5. Verify Results
```bash
echo "=== EMAIL LOGS (should show sent -> delivered -> opened) ==="
curl http://localhost:3000/logs | jq

echo -e "\n=== FILTER LOGS BY EMAIL ==="
curl "http://localhost:3000/logs?email=demo@example.com" | jq

echo -e "\n=== FILTER LOGS BY STATUS ==="
curl "http://localhost:3000/logs?status=delivered" | jq
```

## 🎯 Part 5: Teacher Talking Points

### What This Demonstrates:

1. **RESTful API Design**
   - Clean endpoint structure
   - Proper HTTP methods and status codes
   - JSON request/response format

2. **Webhook Integration**
   - Real-time communication between systems
   - Event-driven architecture
   - External service integration

3. **Data Flow Management**
   - In-memory data storage
   - Data validation and error handling
   - Structured logging

4. **Automation with n8n**
   - Visual workflow creation
   - API orchestration
   - Real-time execution monitoring

5. **Production-Ready Features**
   - CORS support for frontend integration
   - Input validation
   - Error handling and proper responses

## 🔄 Part 6: Additional Demo Scenarios

### Scenario A: Test Error Handling
```bash
# Try invalid email
curl -X POST http://localhost:3000/subscribe \
  -H "Content-Type: application/json" \
  -d '{"email": "invalid-email"}'

# Try duplicate subscription
curl -X POST http://localhost:3000/subscribe \
  -H "Content-Type: application/json" \
  -d '{"email": "demo@example.com"}'
```

### Scenario B: Test Filtering
```bash
# Add more test data
curl -X POST http://localhost:3000/webhook/email-status \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "status": "failed"}'

# Show filtering capabilities
curl "http://localhost:3000/logs?status=failed" | jq
```

### Scenario C: Multiple Workflow Runs
```bash
# Trigger workflow multiple times to show accumulating logs
curl -X POST http://localhost:5678/webhook/demo-newsletter \
  -H "Content-Type: application/json" \
  -d '{"testEmail": "student1@university.edu"}'

sleep 10

curl -X POST http://localhost:5678/webhook/demo-newsletter \
  -H "Content-Type: application/json" \
  -d '{"testEmail": "teacher@university.edu"}'
```

## 📊 Expected Demo Results

After running the demo, you should see:

- **Subscribers:** 3 email addresses
- **Newsletters:** 1 newsletter created
- **Logs:** Multiple entries showing email journey (sent → delivered → opened)
- **n8n Workflow:** Successfully executed with visual feedback

## 🎓 Key Learning Outcomes

1. **Express.js API development** with modern ES6 syntax
2. **Webhook implementation** for real-time integrations
3. **n8n workflow automation** for business processes
4. **Data validation and error handling** best practices
5. **RESTful API design** principles

---

**🎉 Your demo is ready!** This showcases a complete, working integration between your Express.js newsletter system and n8n automation platform.
