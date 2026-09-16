import { useId, cloneElement, isValidElement, type ReactNode, type ReactElement } from 'react';
import { cn } from '../../lib/utils';

export interface FormFieldProps {
    /** Visible label text, associated with the control via htmlFor/id. */
    label: ReactNode;
    /** The single form control (Input, Select, textarea…). Receives an `id`. */
    children: ReactNode;
    /** Muted helper line under the control. */
    help?: ReactNode;
    /** Error line under the control — replaces `help` while present. */
    error?: ReactNode;
    className?: string;
}

/**
 * The one way to lay out a labeled control: label (associated via htmlFor),
 * control, and a help or error line. Passing an element child without an `id`
 * gets the generated one so the pair is always programmatically linked.
 */
export function FormField({ label, children, help, error, className }: FormFieldProps) {
    const generatedId = useId();
    let control = children;
    let controlId = generatedId;
    if (isValidElement(children)) {
        const child = children as ReactElement<{ id?: string }>;
        controlId = child.props.id ?? generatedId;
        control = child.props.id ? child : cloneElement(child, { id: generatedId });
    }
    return (
        <div className={cn('flex flex-col gap-1.5', className)}>
            <label htmlFor={controlId} className="text-label text-fg">
                {label}
            </label>
            {control}
            {error ? (
                <p className="text-caption text-danger">{error}</p>
            ) : (
                help && <p className="text-caption text-fg-muted">{help}</p>
            )}
        </div>
    );
}

export default FormField;
