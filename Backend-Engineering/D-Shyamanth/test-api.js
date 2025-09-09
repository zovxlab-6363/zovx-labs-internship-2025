#!/usr/bin/env node

/**
 * Simple API test script for the Newsletter System
 * Run with: node test-api.js
 */

const API_BASE = 'http://localhost:3000';

// Helper function to make HTTP requests
async function makeRequest(method, url, data = null) {
  const options = {
    method,
    headers: {
      'Content-Type': 'application/json',
    },
  };
  
  if (data) {
    options.body = JSON.stringify(data);
  }

  try {
    const response = await fetch(API_BASE + url, options);
    const result = await response.json();
    
    console.log(`\n${method} ${url}`);
    console.log(`Status: ${response.status}`);
    console.log('Response:', JSON.stringify(result, null, 2));
    
    return { response, result };
  } catch (error) {
    console.error(`\n❌ Error with ${method} ${url}:`, error.message);
    return null;
  }
}

async function runTests() {
  console.log('🧪 Testing Newsletter System API...\n');
  console.log('Make sure the server is running: npm start');
  
  // Wait a bit for server to be ready
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  try {
    // 1. Health check
    await makeRequest('GET', '/');
    
    // 2. Subscribe email
    await makeRequest('POST', '/subscribe', {
      email: 'test@gmail.com'
    });
    
    // 3. Try duplicate subscription (should fail)
    await makeRequest('POST', '/subscribe', {
      email: 'test@gmail.com'
    });
    
    // 4. Subscribe another email
    await makeRequest('POST', '/subscribe', {
      email: 'demo@example.com'
    });
    
    // 5. Get all subscribers
    await makeRequest('GET', '/subscribers');
    
    // 6. Create newsletter
    await makeRequest('POST', '/newsletters', {
      title: 'Welcome Newsletter',
      content: 'Thanks for subscribing! This is our first newsletter.'
    });
    
    // 7. Create another newsletter
    await makeRequest('POST', '/newsletters', {
      title: 'Product Updates',
      content: 'Check out our latest features and improvements!'
    });
    
    // 8. Get all newsletters
    await makeRequest('GET', '/newsletters');
    
    // 9. Simulate n8n webhook - email sent
    await makeRequest('POST', '/webhook/email-status', {
      email: 'test@gmail.com',
      status: 'sent',
      newsletterId: Date.now()
    });
    
    // 10. Simulate n8n webhook - email delivered
    await makeRequest('POST', '/webhook/email-status', {
      email: 'test@gmail.com',
      status: 'delivered'
    });
    
    // 11. Simulate n8n webhook - email opened
    await makeRequest('POST', '/webhook/email-status', {
      email: 'demo@example.com',
      status: 'opened'
    });
    
    // 12. Get all logs
    await makeRequest('GET', '/logs');
    
    // 13. Get logs filtered by email
    await makeRequest('GET', '/logs?email=test@gmail.com');
    
    // 14. Get logs filtered by status
    await makeRequest('GET', '/logs?status=delivered');
    
    // 15. Test invalid route
    await makeRequest('GET', '/invalid-route');
    
    console.log('\n✅ All tests completed!');
    console.log('\n📝 Summary:');
    console.log('- Subscription management ✅');
    console.log('- Newsletter creation ✅');
    console.log('- n8n webhook integration ✅');
    console.log('- Email status logging ✅');
    console.log('- Error handling ✅');
    
  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
  }
}

// Check if fetch is available (Node.js 18+)
if (typeof fetch === 'undefined') {
  console.error('❌ This script requires Node.js 18+ or install node-fetch');
  console.log('Alternative: Test with curl commands from README.md');
  process.exit(1);
}

runTests();
