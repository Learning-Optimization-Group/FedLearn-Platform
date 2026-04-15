import React, { useEffect, useRef } from 'react';
import '../styles/ConfirmDialog.css';

interface ConfirmDialogProps {
    title: string;
    message: string;
    confirmLabel?: string;
    cancelLabel?: string;
    danger?: boolean;
    onConfirm: () => void;
    onCancel: () => void;
}

const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
    title,
    message,
    confirmLabel = 'Confirm',
    cancelLabel = 'Cancel',
    danger = false,
    onConfirm,
    onCancel,
}) => {
    const confirmBtnRef = useRef<HTMLButtonElement>(null);

    useEffect(() => {
        confirmBtnRef.current?.focus();
    }, []);

    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (e.key === 'Escape') onCancel();
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [onCancel]);

    return (
        <div
            className="confirm-dialog-backdrop"
            role="dialog"
            aria-modal="true"
            aria-labelledby="confirm-dialog-title"
            onClick={(e) => {
                if (e.target === e.currentTarget) onCancel();
            }}
        >
            <div className="confirm-dialog" onClick={(e) => e.stopPropagation()}>
                <h3 id="confirm-dialog-title">{title}</h3>
                <p>{message}</p>
                <div className="confirm-dialog-actions">
                    <button type="button" className="btn-cancel" onClick={onCancel}>
                        {cancelLabel}
                    </button>
                    <button
                        ref={confirmBtnRef}
                        type="button"
                        className={danger ? 'btn-danger' : 'btn-primary'}
                        onClick={onConfirm}
                    >
                        {confirmLabel}
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ConfirmDialog;
