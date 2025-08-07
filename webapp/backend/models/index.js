const mongoose = require('mongoose');

const Thread = require('./Thread');
const Project = require('./Project');

// Export all models
module.exports = {
  Thread,
  Project
};

// Connection helper
const connectDB = async (connectionString) => {
  try {
    const conn = await mongoose.connect(connectionString, {
      useNewUrlParser: true,
      useUnifiedTopology: true
    });

    console.log(`MongoDB Connected: ${conn.connection.host}`);
    
    // Create indexes if they don't exist
    await Promise.all([
      Thread.ensureIndexes(),
      Project.ensureIndexes()
    ]);

    return conn;
  } catch (error) {
    console.error('Database connection error:', error);
    process.exit(1);
  }
};

module.exports.connectDB = connectDB;
