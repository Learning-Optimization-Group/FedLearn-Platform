import React from 'react';
import '../styles/DiskLoader.css';

interface DiskLoaderProps {
    message?: string;
}

const DiskLoader: React.FC<DiskLoaderProps> = ({ message = "Loading..." }) => {
    return (
        <div className="disk-loader-container" role="status" aria-live="polite">
            <div className="disk-loader">
                <div className="disk"></div>
                <div className="disk"></div>
                <div className="disk"></div>
            </div>
            {message && <p className="loader-message">{message}</p>}
        </div>
    );
};

export default DiskLoader;
