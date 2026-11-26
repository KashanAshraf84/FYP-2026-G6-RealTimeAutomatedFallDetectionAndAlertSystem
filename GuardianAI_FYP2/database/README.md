# GuardianAI — PostgreSQL database

`guardianai_postgres.sql` creates the full GuardianAI schema (18 tables) plus starter rows: migration history,
the 4 example cameras and the default detection settings. It contains no users, incidents or passwords. It was generated
from the project's real Django migrations and checked by loading it into an empty PostgreSQL 16 database:
no errors, and `manage.py migrate --check` reports nothing missing.

> The live `guardianai` database is **protected** (see `docs/DATABASE.md`). Run the SQL file only against a new,
> empty database. It has no DROP/DELETE statements, so on an existing database it stops with "already exists" errors.

Default names used below (they match `config/settings.py`): database `guardianai`, user `guardian`, host
`localhost`, port `5432`. Pick your own password.

---

## 1. Install PostgreSQL (once)

- **With admin rights:** the installer from https://www.postgresql.org/download/windows/ (includes `psql` and pgAdmin 4).
- **Without admin rights:** download the *zip binaries* from https://www.enterprisedb.com/download-postgresql-binaries,
  then extract them to a folder you own, e.g. `C:\Users\<you>\pg`, so that you have `C:\Users\<you>\pg\pgsql\bin\psql.exe`. Then:

  ```powershell
  $PG = "$env:USERPROFILE\pg"
  & "$PG\pgsql\bin\initdb.exe" -D "$PG\data" -U postgres -E UTF8 -A scram-sha-256 -W   # asks for a 'postgres' password
  & "$PG\pgsql\bin\pg_ctl.exe" -D "$PG\data" -l "$PG\postgres.log" start
  ```
  Set `GUARDIAN_PG_HOME=C:\Users\<you>\pg` in `.env` and `start.ps1` will start this server for you next time.

## 2. Create the login and the database (once)

```powershell
psql -h localhost -U postgres
```
```sql
CREATE ROLE guardian LOGIN PASSWORD 'choose-a-password' CREATEDB;   -- CREATEDB lets `manage.py test` make test_guardianai
CREATE DATABASE guardianai OWNER guardian ENCODING 'UTF8' TEMPLATE template0;
\q
```

## 3. Create the tables — choose ONE

**A. Django (preferred):** put the settings in `.env` (step 4), then `.venv\Scripts\python manage.py migrate` and `manage.py seed_demo`.

**B. The SQL file:**
```powershell
psql -h localhost -U guardian -d guardianai -v ON_ERROR_STOP=1 -f database\guardianai_postgres.sql
```

## 4. Point GuardianAI at it

In `.env` (project root; never commit it):
```text
GUARDIAN_DB=postgres
GUARDIAN_DB_NAME=guardianai
GUARDIAN_DB_USER=guardian
GUARDIAN_DB_PASSWORD=choose-a-password
GUARDIAN_DB_HOST=127.0.0.1
GUARDIAN_DB_PORT=5432
```
Install the driver (`pip install "psycopg[binary]>=3.2"`, included in `requirements-ai.txt`), run the server, and
register the first account (it becomes Administrator).

---

## Open the database with `psql` (SQL Shell)

Start menu **SQL Shell (psql)** (installer), or `C:\Users\<you>\pg\pgsql\bin\psql.exe` (zip). Press Enter for the defaults and type
`guardianai` / `guardian` when asked, or:
```powershell
psql -h localhost -p 5432 -U guardian -d guardianai
```
Useful commands:
```sql
\dt                                   -- list tables
\d monitor_event                      -- columns of one table
SELECT id, name, location, kind, enabled, last_status FROM monitor_camera;
SELECT id, created_at, track_id, confidence, status, is_demo FROM monitor_event ORDER BY created_at DESC LIMIT 20;
SELECT u.username, u.email, p.role, u.is_active FROM auth_user u JOIN monitor_profile p ON p.user_id = u.id;
SELECT created_at, action, detail FROM monitor_audit ORDER BY created_at DESC LIMIT 20;
\q                                    -- quit
```

## Open the database with pgAdmin 4

Servers > right-click > **Register > Server**. *General*: name `GuardianAI`. *Connection*: host `localhost`, port
`5432`, maintenance database `guardianai`, username `guardian`, password. Then go to Databases > guardianai > Schemas > public > Tables,
right-click a table > **View/Edit Data**.

## Open the database with DBeaver

1. Install DBeaver Community from https://dbeaver.io/download/. If you can't install software, use the **Windows zip (portable)** version.
2. **Database > New Database Connection** (plug icon with +) > choose **PostgreSQL** > Next.
3. *Main* tab: Connect by **Host**. Host `localhost`, Port `5432`, Database `guardianai`, Authentication *Database Native*,
   Username `guardian`, Password (tick *Save password* if you want).
4. Click **Test Connection**. The first time, DBeaver offers to **download the PostgreSQL driver**: accept. You should see "Connected".
5. **Finish**. In the Database Navigator expand `guardianai > Databases > guardianai > Schemas > public > Tables`.
   - Double-click a table, then the **Data** tab, to browse rows. The **ER Diagram** tab shows the relationships.
   - **SQL Editor > New SQL script** (Ctrl+]) to run queries such as the ones above (Ctrl+Enter runs one statement).
6. To load `guardianai_postgres.sql` from DBeaver instead of psql (empty database only): right-click the `guardianai` database >
   **SQL Editor > Load SQL script…**, pick the file, then **Execute SQL Script** (Alt+X).

## Moving existing SQLite data into PostgreSQL
See the README "Database" section (`dumpdata` with SQLite, switch `.env`, `migrate`, `loaddata`). Back up first.

## Regenerating the SQL file
After any model or migration change, recreate it from a throwaway database: `manage.py migrate` into an empty database, then
`pg_dump --schema-only --no-owner --no-privileges` plus a `--data-only --inserts` dump of `django_migrations`,
`django_content_type`, `auth_permission`, `monitor_camera` and `monitor_detectionsettings`.
