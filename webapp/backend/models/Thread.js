const mongoose = require('mongoose');

const MessageSchema = new mongoose.Schema({
  content: {
    type: String,
    required: true,
    trim: true
  },
  sender: {
    type: String,
    required: true,
    enum: ['user', 'agent', 'system']
  },
  agentType: {
    type: String,
    default: null,
    enum: [null, 'deduction', 'metaphor', 'planner', 'ritual', 'general']
  },
  metadata: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },
  timestamp: {
    type: Date,
    default: Date.now
  }
}, { _id: true });

const ThreadSchema = new mongoose.Schema({
  title: {
    type: String,
    required: true,
    trim: true,
    maxlength: 200
  },
  userId: {
    type: String,
    required: true,
    default: 'default'
  },
  agentType: {
    type: String,
    default: 'general',
    enum: ['deduction', 'metaphor', 'planner', 'ritual', 'general', 'mixed']
  },
  tags: [{
    type: String,
    trim: true,
    lowercase: true
  }],
  messages: [MessageSchema],
  messageCount: {
    type: Number,
    default: 0
  },
  lastMessage: {
    type: String,
    default: null,
    maxlength: 500
  },
  isArchived: {
    type: Boolean,
    default: false
  },
  settings: {
    autoSave: {
      type: Boolean,
      default: true
    },
    notifications: {
      type: Boolean,
      default: true
    }
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes for performance
ThreadSchema.index({ userId: 1, updatedAt: -1 });
ThreadSchema.index({ userId: 1, isArchived: 1 });
ThreadSchema.index({ tags: 1 });
ThreadSchema.index({ agentType: 1 });

// Virtual for getting recent messages
ThreadSchema.virtual('recentMessages').get(function() {
  return this.messages.slice(-10);
});

// Instance method to add a message
ThreadSchema.methods.addMessage = function(messageData) {
  this.messages.push(messageData);
  this.messageCount = this.messages.length;
  this.lastMessage = messageData.content.substring(0, 100);
  this.updatedAt = new Date();
  return this.save();
};

// Instance method to get conversation summary
ThreadSchema.methods.getSummary = function() {
  const userMessages = this.messages.filter(msg => msg.sender === 'user').length;
  const agentMessages = this.messages.filter(msg => msg.sender === 'agent').length;
  
  return {
    id: this._id,
    title: this.title,
    totalMessages: this.messageCount,
    userMessages,
    agentMessages,
    primaryAgent: this.agentType,
    tags: this.tags,
    lastActivity: this.updatedAt,
    isArchived: this.isArchived
  };
};

// Static method to find threads by agent type
ThreadSchema.statics.findByAgentType = function(agentType, userId = 'default') {
  return this.find({ agentType, userId, isArchived: false })
    .sort({ updatedAt: -1 })
    .select('-messages');
};

// Static method to search threads
ThreadSchema.statics.searchThreads = function(query, userId = 'default') {
  const searchRegex = new RegExp(query, 'i');
  
  return this.find({
    userId,
    $or: [
      { title: searchRegex },
      { tags: { $in: [searchRegex] } },
      { lastMessage: searchRegex }
    ]
  }).sort({ updatedAt: -1 });
};

module.exports = mongoose.model('Thread', ThreadSchema);
