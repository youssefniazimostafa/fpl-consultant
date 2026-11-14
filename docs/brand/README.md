# XPert brand

Original marks for XPert, your fantasy consultant (no league, club, UEFA or FPL artwork).

- `xpert-logo.svg`: full logo, a rounded tile with faint pitch markings (both penalty boxes, a
  halfway line that stops at the centre circle) and "xP".
- `xpert-icon.svg`: simplified icon for small sizes (tile, centre circle, halfway-line stubs, "xP").
- `xpert-wordmark.svg`: the icon, "XPert" and "your fantasy consultant".
- `xpert-app-icon.svg` / `xpert-app-icon-mac.svg`: the simplified icon on a solid dark tile, the
  sources of the iPhone / web icons and the Mac app icon (`app/scripts/make-icons.sh`).

Font: Barlow Condensed Bold (SIL Open Font License 1.1), bundled with the app through
`@fontsource/barlow-condensed`; used only for the logo and wordmark. The SVGs reference it by
name, so view them with the font installed (or through the app). In the app the marks are the
React components in `app/src/components/Brand.tsx`, drawn in the surrounding text colour.
