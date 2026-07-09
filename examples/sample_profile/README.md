# Sample profile

An example app profile for the bundled **demo target app**.

## Try it

1. Copy this folder into your workspace (keep the folder name — it is the profile id):

   ```bash
   mkdir -p ~/LocalRPAStudio/profiles
   cp -R examples/sample_profile ~/LocalRPAStudio/profiles/
   ```

2. Start the demo target app in one terminal:

   ```bash
   rpa-demo-app
   ```

3. Start Local RPA Studio in another terminal:

   ```bash
   local-rpa-studio
   ```

4. Select the `sample_profile` profile, select the `demo_submit_form` workflow,
   and click **Dry Run** to preview matches, then **▶ Run** for real.

## Important: re-capture the template on your own screen

The bundled `targets/submit_button.png` was rendered synthetically, so it may
not match your screen's fonts, theme, or DPI well. If the dry run reports low
confidence, use **Capture Target…** to grab the Submit button from *your*
screen and either give it the id `submit_button` (after deleting the bundled
one) or update the workflow's `target_id`.

This works with any visible UI element — a button on a web page in your
browser works just as well as the demo app.
