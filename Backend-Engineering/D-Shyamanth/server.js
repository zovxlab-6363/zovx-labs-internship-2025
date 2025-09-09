import express from 'express';
import cors from 'cors';

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 3000;

// In-memory storage (no database required)
const subscribers = [];
const newsletters = [];
const emailLogs = [];

// Middleware
app.use(express.json());
app.use(cors());

// Error handling middleware
const errorHandler = (err, req, res, next) => {
  console.error('Error:', err.message);
  res.status(500).json({
    success: false,
    message: 'Internal server error',
    error: err.message
  });
};

// Validation middleware for email
const validateEmail = (email) => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

// Routes

/**
 * POST /subscribe
 * Subscribe a new email to the newsletter
 * Body: { "email": "user@example.com" }
 */
app.post('/subscribe', (req, res) => {
  try {
    const { email } = req.body;

    // Validate input
    if (!email) {
      return res.status(400).json({
        success: false,
        message: 'Email is required'
      });
    }

    if (!validateEmail(email)) {
      return res.status(400).json({
        success: false,
        message: 'Please provide a valid email address'
      });
    }

    // Check if email already exists
    if (subscribers.find(sub => sub.email === email)) {
      return res.status(409).json({
        success: false,
        message: 'Email already subscribed'
      });
    }

    // Add subscriber
    const subscriber = {
      id: Date.now(),
      email: email.toLowerCase(),
      subscribedAt: new Date().toISOString()
    };

    subscribers.push(subscriber);

    res.status(201).json({
      success: true,
      message: 'Successfully subscribed!',
      data: subscriber
    });

  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Failed to subscribe',
      error: error.message
    });
  }
});

/**
 * GET /subscribers
 * Get all subscribers
 */
app.get('/subscribers', (req, res) => {
  try {
    res.json({
      success: true,
      count: subscribers.length,
      data: subscribers
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Failed to fetch subscribers',
      error: error.message
    });
  }
});

/**
 * POST /newsletters
 * Create a new newsletter
 * Body: { "title": "Newsletter Title", "content": "Newsletter content..." }
 */
app.post('/newsletters', (req, res) => {
  try {
    const { title, content } = req.body;

    // Validate input
    if (!title || !content) {
      return res.status(400).json({
        success: false,
        message: 'Title and content are required'
      });
    }

    // Create newsletter
    const newsletter = {
      id: Date.now(),
      title: title.trim(),
      content: content.trim(),
      createdAt: new Date().toISOString(),
      sentTo: 0 // Track how many people it was sent to
    };

    newsletters.push(newsletter);

    res.status(201).json({
      success: true,
      message: 'Newsletter created successfully!',
      data: newsletter
    });

  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Failed to create newsletter',
      error: error.message
    });
  }
});

/**
 * GET /newsletters
 * Get all newsletters
 */
app.get('/newsletters', (req, res) => {
  try {
    res.json({
      success: true,
      count: newsletters.length,
      data: newsletters
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Failed to fetch newsletters',
      error: error.message
    });
  }
});

/**
 * POST /webhook/email-status
 * Webhook endpoint for n8n to report email delivery status
 * Body: { "email": "user@example.com", "status": "sent|failed|delivered|bounced" }
 */
app.post('/webhook/email-status', (req, res) => {
  try {
    const { email, status, newsletterId } = req.body;

    // Validate input
    if (!email || !status) {
      return res.status(400).json({
        success: false,
        message: 'Email and status are required'
      });
    }

    // Validate email format
    if (!validateEmail(email)) {
      return res.status(400).json({
        success: false,
        message: 'Please provide a valid email address'
      });
    }

    // Validate status
    const validStatuses = ['sent', 'delivered', 'failed', 'bounced', 'opened', 'clicked'];
    if (!validStatuses.includes(status.toLowerCase())) {
      return res.status(400).json({
        success: false,
        message: `Invalid status. Valid statuses: ${validStatuses.join(', ')}`
      });
    }

    // Create log entry
    const logEntry = {
      id: Date.now(),
      email: email.toLowerCase(),
      status: status.toLowerCase(),
      newsletterId: newsletterId || null,
      timestamp: new Date().toISOString(),
      receivedAt: new Date().toISOString()
    };

    emailLogs.push(logEntry);

    res.status(200).json({
      success: true,
      message: 'Email status logged successfully',
      data: logEntry
    });

  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Failed to log email status',
      error: error.message
    });
  }
});

/**
 * GET /logs
 * Get all email delivery logs
 */
app.get('/logs', (req, res) => {
  try {
    // Optional query parameters for filtering
    const { email, status } = req.query;
    
    let filteredLogs = [...emailLogs];

    if (email) {
      filteredLogs = filteredLogs.filter(log => 
        log.email.toLowerCase().includes(email.toLowerCase())
      );
    }

    if (status) {
      filteredLogs = filteredLogs.filter(log => 
        log.status.toLowerCase() === status.toLowerCase()
      );
    }

    // Sort by most recent first
    filteredLogs.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

    res.json({
      success: true,
      count: filteredLogs.length,
      totalLogs: emailLogs.length,
      data: filteredLogs
    });

  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Failed to fetch logs',
      error: error.message
    });
  }
});

/**
 * GET /
 * Health check and API info
 */
app.get('/', (req, res) => {
  res.json({
    success: true,
    message: 'Newsletter System API is running!',
    version: '1.0.0',
    endpoints: {
      'POST /subscribe': 'Subscribe to newsletter',
      'GET /subscribers': 'Get all subscribers',
      'POST /newsletters': 'Create newsletter',
      'GET /newsletters': 'Get all newsletters',
      'POST /webhook/email-status': 'n8n webhook for email status',
      'GET /logs': 'Get email delivery logs'
    },
    stats: {
      subscribers: subscribers.length,
      newsletters: newsletters.length,
      logs: emailLogs.length
    }
  });
});

// Handle 404 routes
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    message: 'Route not found',
    availableEndpoints: [
      'GET /',
      'POST /subscribe',
      'GET /subscribers', 
      'POST /newsletters',
      'GET /newsletters',
      'POST /webhook/email-status',
      'GET /logs'
    ]
  });
});

// Apply error handling middleware
app.use(errorHandler);

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Newsletter System API running on http://localhost:${PORT}`);
  console.log(`📧 Ready to accept subscriptions and send newsletters!`);
  console.log(`🔗 n8n webhook endpoint: http://localhost:${PORT}/webhook/email-status`);
});

// Export app for testing purposes
export default app;
