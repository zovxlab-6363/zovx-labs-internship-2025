# Newsletter System API

A simple Express.js newsletter system with n8n webhook integration for zovx purposes.

## Features

- ✅ Email subscription management
- ✅ Newsletter creation and storage
- ✅ n8n webhook integration for email status tracking
- ✅ In-memory storage (no database required)
- ✅ Input validation and error handling
- ✅ Clean REST API design

## Quick Start

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start the server:**
   ```bash
   npm start
   ```
   
   Or for development with auto-restart:
   ```bash
   npm run dev
   ```

3. **API will be running at:** `http://localhost:3000`

## API Endpoints

### 1. Health Check
```http
GET /
```
Returns API status and statistics.

### 2. Subscribe to Newsletter
```http
POST /subscribe
Content-Type: application/json

{
  "email": "test@gmail.com"
}
```

### 3. Get All Subscribers
```http
GET /subscribers
```

### 4. Create Newsletter
```http
POST /newsletters
Content-Type: application/json

{
  "title": "Welcome Newsletter",
  "content": "Thanks for subscribing to our newsletter!"
}
```

### 5. Get All Newsletters
```http
GET /newsletters
```

### 6. n8n Email Status Webhook
```http
POST /webhook/email-status
Content-Type: application/json

{
  "email": "test@gmail.com",
  "status": "sent",
  "newsletterId": 1234567890
}
```

Valid status values: `sent`, `delivered`, `failed`, `bounced`, `opened`, `clicked`

### 7. Get Email Logs
```http
GET /logs
```

Optional query parameters:
- `?email=test@gmail.com` - filter by email
- `?status=sent` - filter by status

## Example Test Flow

1. **Subscribe an email:**
   ```bash
   curl -X POST http://localhost:3000/subscribe \
     -H "Content-Type: application/json" \
     -d '{"email": "test@gmail.com"}'
   ```

2. **Create a newsletter:**
   ```bash
   curl -X POST http://localhost:3000/newsletters \
     -H "Content-Type: application/json" \
     -d '{"title": "Hello World", "content": "Welcome to our newsletter!"}'
   ```

3. **Simulate n8n webhook (email sent):**
   ```bash
   curl -X POST http://localhost:3000/webhook/email-status \
     -H "Content-Type: application/json" \
     -d '{"email": "test@gmail.com", "status": "sent"}'
   ```

4. **Check logs:**
   ```bash
   curl http://localhost:3000/logs
   ```

## Response Format

All endpoints return JSON in this format:
```json
{
  "success": true|false,
  "message": "Description",
  "data": {...},
  "count": 123
}
```

## Error Handling

- ✅ Input validation
- ✅ Duplicate email prevention
- ✅ Proper HTTP status codes
- ✅ Descriptive error messages

## Data Storage

All data is stored in memory using JavaScript arrays:
- `subscribers[]` - Email subscriptions
- `newsletters[]` - Created newsletters  
- `emailLogs[]` - Email delivery status logs

**Note:** Data will be lost when the server restarts (perfect for demo purposes).

## n8n Integration

Configure your n8n workflow to send POST requests to:
```
http://localhost:3000/webhook/email-status
```

With payload:
```json
{
  "email": "recipient@example.com",
  "status": "sent|delivered|failed|bounced|opened|clicked",
  "newsletterId": 1234567890
}
```
