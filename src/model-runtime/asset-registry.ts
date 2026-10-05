const assetFiles = import.meta.glob<string>(
  "../modules/*/assets/*",
  {
    eager: true,
    query: "?url",
    import: "default",
  },
);

export function teachingAssetUrl(moduleSlug: string, filename: string) {
  const key = `../modules/${moduleSlug}/assets/${filename}`;
  const url = assetFiles[key];
  if (!url) {
    throw new Error(`Unknown teaching asset: ${moduleSlug}/${filename}`);
  }
  return url;
}

export const teachingAssetCount = Object.keys(assetFiles).length;
