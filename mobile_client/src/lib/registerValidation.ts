// MO-6: register-flow input validation, extracted out of RegisterScreen so it is unit-testable without
// rendering the screen (the mobile convention keeps screen logic in libs and RTL-renders nothing). The
// rules are exactly what the screen enforced inline: username >= 3 chars (trimmed), a basic email shape,
// password >= 6 chars (NOT trimmed — spaces are legal password characters). On success it returns the
// CLEANED values (trimmed username/email, verbatim password) so the caller submits the same normalized
// form it validated, instead of re-trimming at the call site.

export interface RegistrationInput {
  username: string;
  email: string;
  password: string;
}

export type RegistrationCheck =
  | { ok: true; username: string; email: string; password: string }
  | { ok: false; error: string };

// Deliberately permissive: a real address check happens server-side; this only rejects the obvious
// "that isn't an email" typo (no '@', no dot-domain, or embedded whitespace) before a network call.
const EMAIL_RE = /^\S+@\S+\.\S+$/;

export const MIN_USERNAME_LEN = 3;
export const MIN_PASSWORD_LEN = 6;

export function validateRegistration(input: RegistrationInput): RegistrationCheck {
  const username = input.username.trim();
  const email = input.email.trim();
  const password = input.password; // NOT trimmed — leading/trailing spaces are valid password chars.

  if (username.length < MIN_USERNAME_LEN) {
    return { ok: false, error: `Username must be at least ${MIN_USERNAME_LEN} characters.` };
  }
  if (!EMAIL_RE.test(email)) {
    return { ok: false, error: 'Enter a valid email address.' };
  }
  if (password.length < MIN_PASSWORD_LEN) {
    return { ok: false, error: `Password must be at least ${MIN_PASSWORD_LEN} characters.` };
  }
  return { ok: true, username, email, password };
}
