import { useCallback, useEffect, useState } from "react";
import {
  addSample,
  createPerson,
  deletePerson,
  deleteSample,
  listPeople,
  listSamples,
  loadDemo,
  removeDemo,
  sampleImageUrl,
  type Box,
  type Person,
  type Sample,
} from "../api/client";
import BoxDrawer from "../components/BoxDrawer";

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function PeopleView() {
  const [people, setPeople] = useState<Person[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [name, setName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [box, setBox] = useState<Box | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const list = await listPeople();
    setPeople(list);
    return list;
  }, []);

  useEffect(() => {
    refresh().catch((e) => setError(errorText(e)));
  }, [refresh]);

  useEffect(() => {
    if (selected === null) return setSamples([]);
    listSamples(selected).then(setSamples, (e) => setError(errorText(e)));
  }, [selected]);

  /** Run an action with a busy flag and inline error reporting. */
  async function run(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (e) {
      setError(errorText(e));
    } finally {
      setBusy(false);
    }
  }

  const person = people.find((p) => p.id === selected) ?? null;
  const hasDemo = people.some((p) => p.is_demo);

  return (
    <div className="view">
      <div className="cols">
        <section className="card" aria-label="People">
          <h2>People</h2>
          <form
            className="row"
            onSubmit={(e) => {
              e.preventDefault();
              run(async () => {
                const created = await createPerson(name);
                setName("");
                await refresh();
                setSelected(created.id);
              });
            }}
          >
            <input
              id="new-person"
              type="text"
              placeholder="Name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              aria-label="New person's name"
            />
            <button className="btn primary" disabled={busy || !name.trim()}>
              Add
            </button>
          </form>
          <ul className="people">
            {people.map((p) => (
              <li key={p.id}>
                <button aria-current={p.id === selected} onClick={() => setSelected(p.id)}>
                  <span>{p.name}</span>
                  <span className="row">
                    {p.is_demo && <span className="badge">demo</span>}
                    <span className="small muted">{p.n_samples}</span>
                  </span>
                </button>
              </li>
            ))}
            {people.length === 0 && <li className="small muted">No one enrolled yet.</li>}
          </ul>
          <div className="row">
            {hasDemo ? (
              <button
                className="btn danger"
                disabled={busy}
                onClick={() => run(async () => {
                  await removeDemo();
                  setSelected(null);
                  await refresh();
                })}
              >
                Remove ORL demo
              </button>
            ) : (
              <button className="btn" disabled={busy} onClick={() => run(async () => {
                await loadDemo();
                await refresh();
              })}>
                {busy ? "Loading…" : "Load ORL demo (30 people)"}
              </button>
            )}
          </div>
          <p className="small muted">
            Demo photos: ORL Database of Faces, credit Olivetti Research Laboratory (AT&amp;T
            Laboratories Cambridge). Downloaded once, kept on this machine.
          </p>
        </section>

        <section className="card" aria-label="Face samples">
          {person === null ? (
            <p className="muted">Select or add a person to see and add their face photos.</p>
          ) : (
            <>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h2>{person.name}</h2>
                <button
                  className="btn danger"
                  disabled={busy}
                  onClick={() => run(async () => {
                    await deletePerson(person.id);
                    setSelected(null);
                    await refresh();
                  })}
                >
                  Delete person
                </button>
              </div>
              <div className="faces">
                {samples.map((s) => (
                  <figure key={s.id}>
                    <img className="face" src={sampleImageUrl(s.id)} alt={`${person.name}, sample ${s.id}`}
                      style={{ objectFit: "contain" }} />
                    <figcaption>
                      <span>#{s.id}</span>
                      <button className="btn danger small" style={{ padding: "0 6px" }} aria-label="Delete sample"
                        onClick={() => run(async () => {
                          await deleteSample(s.id);
                          setSamples(await listSamples(person.id));
                          await refresh();
                        })}>
                        ×
                      </button>
                    </figcaption>
                  </figure>
                ))}
              </div>
              {!person.is_demo && (
                <div className="card" style={{ background: "var(--bg)" }}>
                  <h3>Add a photo</h3>
                  <input
                    id="sample-file"
                    type="file"
                    accept="image/*"
                    onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                  />
                  {file && <BoxDrawer file={file} onChange={setBox} />}
                  <div className="row">
                    <button
                      className="btn primary"
                      disabled={busy || !file || !box}
                      onClick={() => run(async () => {
                        if (!file || !box) return;
                        await addSample(person.id, file, box);
                        setFile(null);
                        setSamples(await listSamples(person.id));
                        await refresh();
                      })}
                    >
                      Save face
                    </button>
                    <span className="small muted">Frontal, well lit, several photos per person.</span>
                  </div>
                </div>
              )}
            </>
          )}
          {error && <p className="error" role="alert">{error}</p>}
        </section>
      </div>
    </div>
  );
}
