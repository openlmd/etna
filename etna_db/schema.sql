drop table if exists simpsons;
create table simpsons (
  simpsons_id integer primary key autoincrement,
  first_name text not null,
  last_name text not null
);
