import React, { useState } from 'react';
import '../styles/CopyIcon.css';

// Two SVGs for the copy and checkmark states
const IconCopy = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
    </svg>
);
const IconCheck = () => (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#28a745" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="20 6 9 17 4 12"></polyline>
    </svg>
);

const CopyIcon = ({ textToCopy }) => {
    const [isCopied, setIsCopied] = useState(false);

    const handleCopy = () => {
        // Use the modern navigator.clipboard API
        navigator.clipboard.writeText(textToCopy).then(() => {
            setIsCopied(true);
            // Reset the icon back to the copy state after 2 seconds
            setTimeout(() => setIsCopied(false), 2000);
        }).catch(err => {
            console.error('Failed to copy text: ', err);
            // You could add user feedback here, e.g., an alert
        });
    };

    return (
        <button onClick={handleCopy} className="copy-icon-btn" title="Copy to clipboard">
            {isCopied ? <IconCheck /> : <IconCopy />}
        </button>
    );
};

export default CopyIcon;