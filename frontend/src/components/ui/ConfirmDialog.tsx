import { Modal } from './Modal';
import { Button } from './Button';

export interface ConfirmDialogProps {
    open: boolean;
    title: string;
    message: string;
    confirmLabel?: string;
    cancelLabel?: string;
    /** Use the danger styling on the confirm button. */
    danger?: boolean;
    onConfirm: () => void;
    onCancel: () => void;
}

/**
 * The shared replacement for window.confirm — built on Modal. Confirm button is
 * `primary`, or `danger` when `danger` is set. Cancel is `secondary`.
 */
export function ConfirmDialog({
    open,
    title,
    message,
    confirmLabel = 'Confirm',
    cancelLabel = 'Cancel',
    danger = false,
    onConfirm,
    onCancel,
}: ConfirmDialogProps) {
    return (
        <Modal
            open={open}
            onClose={onCancel}
            title={title}
            size="sm"
            showClose={false}
            footer={
                <>
                    <Button variant="secondary" onClick={onCancel}>
                        {cancelLabel}
                    </Button>
                    <Button variant={danger ? 'danger' : 'primary'} onClick={onConfirm}>
                        {confirmLabel}
                    </Button>
                </>
            }
        >
            <p className="text-body text-fg-muted">{message}</p>
        </Modal>
    );
}

export default ConfirmDialog;
