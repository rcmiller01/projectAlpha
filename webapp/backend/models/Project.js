const mongoose = require('mongoose');

const ProjectSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    trim: true,
    unique: true,
    maxlength: 100
  },
  description: {
    type: String,
    trim: true,
    maxlength: 500
  },
  userId: {
    type: String,
    required: true,
    default: 'default'
  },
  type: {
    type: String,
    required: true,
    enum: ['research', 'creative', 'analysis', 'planning', 'general'],
    default: 'general'
  },
  status: {
    type: String,
    enum: ['active', 'paused', 'completed', 'archived'],
    default: 'active'
  },
  priority: {
    type: String,
    enum: ['low', 'medium', 'high', 'urgent'],
    default: 'medium'
  },
  tags: [{
    type: String,
    trim: true,
    lowercase: true
  }],
  threadIds: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Thread'
  }],
  threadCount: {
    type: Number,
    default: 0
  },
  preferredAgents: [{
    type: String,
    enum: ['deduction', 'metaphor', 'planner', 'ritual', 'general']
  }],
  settings: {
    autoOrganize: {
      type: Boolean,
      default: true
    },
    notifications: {
      type: Boolean,
      default: true
    },
    visibility: {
      type: String,
      enum: ['private', 'shared'],
      default: 'private'
    }
  },
  metadata: {
    totalMessages: {
      type: Number,
      default: 0
    },
    lastActivity: {
      type: Date,
      default: null
    },
    completionProgress: {
      type: Number,
      min: 0,
      max: 100,
      default: 0
    }
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes for performance
ProjectSchema.index({ userId: 1, status: 1, updatedAt: -1 });
ProjectSchema.index({ type: 1 });
ProjectSchema.index({ tags: 1 });
ProjectSchema.index({ priority: 1 });

// Virtual for active threads
ProjectSchema.virtual('activeThreads', {
  ref: 'Thread',
  localField: 'threadIds',
  foreignField: '_id',
  match: { isArchived: false }
});

// Instance method to add a thread
ProjectSchema.methods.addThread = function(threadId) {
  if (!this.threadIds.includes(threadId)) {
    this.threadIds.push(threadId);
    this.threadCount = this.threadIds.length;
    this.metadata.lastActivity = new Date();
    return this.save();
  }
  return Promise.resolve(this);
};

// Instance method to remove a thread
ProjectSchema.methods.removeThread = function(threadId) {
  this.threadIds = this.threadIds.filter(id => !id.equals(threadId));
  this.threadCount = this.threadIds.length;
  this.metadata.lastActivity = new Date();
  return this.save();
};

// Instance method to update progress
ProjectSchema.methods.updateProgress = function(progress) {
  this.metadata.completionProgress = Math.max(0, Math.min(100, progress));
  if (progress >= 100) {
    this.status = 'completed';
  }
  return this.save();
};

// Instance method to get project summary
ProjectSchema.methods.getSummary = function() {
  return {
    id: this._id,
    name: this.name,
    description: this.description,
    type: this.type,
    status: this.status,
    priority: this.priority,
    threadCount: this.threadCount,
    tags: this.tags,
    progress: this.metadata.completionProgress,
    lastActivity: this.metadata.lastActivity || this.updatedAt,
    preferredAgents: this.preferredAgents
  };
};

// Static method to find projects by status
ProjectSchema.statics.findByStatus = function(status, userId = 'default') {
  return this.find({ status, userId })
    .sort({ priority: -1, updatedAt: -1 })
    .select('-threadIds');
};

// Static method to find projects by type
ProjectSchema.statics.findByType = function(type, userId = 'default') {
  return this.find({ type, userId, status: { $ne: 'archived' } })
    .sort({ updatedAt: -1 });
};

// Static method to search projects
ProjectSchema.statics.searchProjects = function(query, userId = 'default') {
  const searchRegex = new RegExp(query, 'i');
  
  return this.find({
    userId,
    status: { $ne: 'archived' },
    $or: [
      { name: searchRegex },
      { description: searchRegex },
      { tags: { $in: [searchRegex] } }
    ]
  }).sort({ updatedAt: -1 });
};

// Static method to get dashboard stats
ProjectSchema.statics.getDashboardStats = function(userId = 'default') {
  return Promise.all([
    this.countDocuments({ userId, status: 'active' }),
    this.countDocuments({ userId, status: 'completed' }),
    this.countDocuments({ userId, status: 'paused' }),
    this.aggregate([
      { $match: { userId } },
      { $group: { _id: '$type', count: { $sum: 1 } } }
    ])
  ]).then(([active, completed, paused, byType]) => ({
    active,
    completed,
    paused,
    total: active + completed + paused,
    byType: byType.reduce((acc, item) => {
      acc[item._id] = item.count;
      return acc;
    }, {})
  }));
};

module.exports = mongoose.model('Project', ProjectSchema);
