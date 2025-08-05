import React from 'react';

const AnchorSettingsPanel = () => {
  return (
    <div style={{ padding: '1rem', border: '1px solid #ccc', borderRadius: '8px' }}>
      <h2>Anchor AI Settings</h2>
      <p>This panel is for configuring the emotional Anchor AI module.</p>
      <ul>
        <li><strong>Memory Limit:</strong> 16GB</li>
        <li><strong>Persona:</strong> Centered Emotional Core</li>
      </ul>
    </div>
  );
};

export default AnchorSettingsPanel;
