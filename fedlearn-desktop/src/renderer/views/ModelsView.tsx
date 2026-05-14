import React from 'react';

const ModelsView: React.FC = () => {
  return (
    <>
      <div className="view-header">
        <div>
          <div className="view-header__title">Models</div>
          <div className="view-header__subtitle">Trained models available to use</div>
        </div>
      </div>
      <div className="placeholder-panel">
        <span className="placeholder-panel__chip">Coming Soon</span>
        <div className="placeholder-panel__title" style={{ marginTop: 14 }}>Model Hub is on the way</div>
        <div className="placeholder-panel__desc">
          Browse, download, and run trained models from federations you participated in — straight from the desktop client.
          For now, models published by a project owner can be downloaded from the web app.
        </div>
      </div>
    </>
  );
};

export default ModelsView;
