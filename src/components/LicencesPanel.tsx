import { Search, TriangleAlert } from "lucide-react";
import { useId, useMemo, useState } from "react";
import {
  APP_COPYRIGHT,
  DELIVERY_LABEL,
  LICENCE_GROUPS,
  STATUS_LABEL,
  additionalNotices,
  filterLicenceEntries,
  licenceEntries,
  licenceTextSections,
  type LicenceEntry,
} from "@app/lib/licenses";

const CHECK_UPSTREAM = "Check the upstream project";

function LicenceRow({ entry }: { entry: LicenceEntry }) {
  const showFlag =
    entry.status === "declared" || (entry.status === "check-upstream" && entry.licence !== CHECK_UPSTREAM);
  return (
    <li className={`licence-entry${entry.secondary ? " licence-entry--compact" : ""}`} data-status={entry.status}>
      <div className="licence-entry__head">
        <strong>{entry.name}</strong>
        {entry.version && <span className="licence-entry__version">{entry.version}</span>}
        <span className="licence-entry__kind">
          {entry.kind} · {DELIVERY_LABEL[entry.delivery]}
        </span>
      </div>
      <p className="licence-entry__use">{entry.usedFor}</p>
      <dl className="licence-entry__facts">
        <div>
          <dt>Licence</dt>
          <dd>
            <span className="licence-badge" data-status={entry.status}>
              {entry.status === "check-upstream" && <TriangleAlert aria-hidden="true" />}
              {entry.licence}
            </span>
          </dd>
        </div>
        <div>
          <dt>Copyright</dt>
          <dd>{entry.holder}</dd>
        </div>
        {entry.url && (
          <div>
            <dt>Source</dt>
            <dd>
              <a href={entry.url} target="_blank" rel="noreferrer">
                {entry.url}
              </a>
            </dd>
          </div>
        )}
      </dl>
      {entry.note && <p className="licence-entry__note">{entry.note}</p>}
      {showFlag && <p className="licence-entry__flag">{STATUS_LABEL[entry.status]}.</p>}
      {!entry.secondary && <p className="licence-entry__basis">Checked against: {entry.basis}.</p>}
    </li>
  );
}

/**
 * About > Licences and attributions: every third-party library, font, runtime, model, and dataset
 * the app bundles or downloads, filterable, with the full licence texts and copyright notices
 * bundled so it all reads offline.
 */
export function LicencesPanel() {
  const [query, setQuery] = useState("");
  const inputId = useId();
  const countId = useId();
  const matches = useMemo(() => filterLicenceEntries(licenceEntries, query), [query]);
  const sections = useMemo(() => licenceTextSections(), []);
  const filtering = query.trim().length > 0;

  return (
    <section className="licences" aria-labelledby="licences-title">
      <h3 id="licences-title">Licences and attributions</h3>
      <p className="licences__intro">
        Discover AI is built on the software, fonts, models, and data below. Each entry says what it is used for, who
        holds the copyright, and under which licence. Entries marked “Downloaded on request” are not part of the
        installer.
      </p>
      <p className="licences__copyright">Discover AI itself: {APP_COPYRIGHT}</p>

      <div className="licences__search">
        <label htmlFor={inputId}>Filter the list</label>
        <div>
          <Search aria-hidden="true" />
          <input
            id={inputId}
            type="search"
            value={query}
            placeholder="Name, licence, or what it is used for"
            autoComplete="off"
            spellCheck={false}
            aria-describedby={countId}
            onChange={(event) => setQuery(event.target.value)}
          />
        </div>
      </div>
      <p id={countId} className="licences__count" role="status">
        {filtering
          ? `${matches.length} of ${licenceEntries.length} components match “${query.trim()}”.`
          : `${licenceEntries.length} components in four groups.`}
      </p>

      {LICENCE_GROUPS.map((group) => {
        const inGroup = matches.filter((entry) => entry.group === group.id);
        if (!inGroup.length) return null;
        const secondary = group.secondaryTitle ? inGroup.filter((entry) => entry.secondary) : [];
        const primary = inGroup.filter((entry) => !secondary.includes(entry));
        const headingId = `licences-group-${group.id}`;
        return (
          <section className="licence-group" key={group.id} aria-labelledby={headingId}>
            <h4 id={headingId}>
              {group.title} <span className="licence-group__count">{inGroup.length}</span>
            </h4>
            <p className="licence-group__blurb">{group.blurb}</p>
            {primary.length > 0 && (
              <ul className="licence-list">
                {primary.map((entry) => (
                  <LicenceRow entry={entry} key={entry.id} />
                ))}
              </ul>
            )}
            {secondary.length > 0 && (
              <details className="licence-more" open={filtering || undefined}>
                <summary>
                  {group.secondaryTitle} ({secondary.length})
                </summary>
                <ul className="licence-list licence-list--compact">
                  {secondary.map((entry) => (
                    <LicenceRow entry={entry} key={entry.id} />
                  ))}
                </ul>
              </details>
            )}
          </section>
        );
      })}

      {filtering && matches.length === 0 && (
        <p className="licences__empty">Nothing matches that. Try a name, a licence such as MIT, or a word like font.</p>
      )}

      <details className="licence-texts">
        <summary>Full licence texts and copyright notices</summary>
        <p>
          Each licence text appears once, below the copyright lines of the components that use it, exactly as their own
          licence files state them. Licences without a text here (such as the Public Domain Dedication and License used
          by GloVe) are linked from their entries.
        </p>
        {sections.map((section) => (
          <details className="licence-text" key={section.id}>
            <summary>
              {section.title} <span>({section.componentCount} components)</span>
            </summary>
            {section.notices.length > 0 && (
              <>
                <h5>Copyright notices</h5>
                <ul className="licence-notices">
                  {section.notices.map((notice) => (
                    <li key={notice.line}>
                      <span>{notice.line}</span> <small>{notice.names.join(", ")}</small>
                    </li>
                  ))}
                </ul>
              </>
            )}
            <h5>Licence text</h5>
            <pre tabIndex={0} role="region" aria-label={`${section.title}, full text`}>
              {section.text}
            </pre>
          </details>
        ))}
        {additionalNotices.length > 0 && (
          <details className="licence-text">
            <summary>
              Notices that differ from the standard texts <span>({additionalNotices.length})</span>
            </summary>
            {additionalNotices.map((notice) => (
              <div key={notice.packages.join(",")}>
                <h5>{notice.packages.join(", ")}</h5>
                <pre tabIndex={0} role="region" aria-label={`Licence notice of ${notice.packages.join(", ")}`}>
                  {notice.text}
                </pre>
              </div>
            ))}
          </details>
        )}
      </details>

      <p className="licences__cloud">
        Cloud providers are not part of this list. If you connect one in Settings, your questions go to that service
        with your own API key, under its own terms. Nothing from a provider is bundled or downloaded.
      </p>
    </section>
  );
}
