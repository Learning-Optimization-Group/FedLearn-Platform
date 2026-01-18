import React from 'react';
import '../styles/DiskLoader.css';

const DiskLoader = ({ message = "Loading..." }) => {
    return (
        <div className="disk-loader-container">
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