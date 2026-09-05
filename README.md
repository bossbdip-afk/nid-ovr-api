Backend v14.6.13 - Step 3 only

- Present Address postal code is rendered with Bengali digits on the card/output.
- Example: 2020 -> ২০২০, 2140 -> ২১৪০.
- Source/debug postalCode remains source-true and is not rewritten.
- Permanent Address, Bengali spacing, address fallback/format, and unrelated fields were not changed in this step.

## Admin login
Set this Render environment variable before using `admin.html`:

`ADMIN_TEMPLATE_KEY=<your-private-admin-key>`

The frontend verifies this key through `POST /admin/login`; the key is kept in browser session storage only and is also required for template save/delete operations.
