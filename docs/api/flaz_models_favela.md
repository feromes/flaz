# flaz.models.favela

## Class Favela

*Sem docstring.*

### Method Favela.__init__

```python
Favela.__init__()
```

Initialize self.  See help(type(self)) for accurate signature.

### Method Favela.calc_hag

```python
Favela.calc_hag()
```

*Sem docstring.*

### Method Favela.calc_vielas

```python
Favela.calc_vielas()
```

*Sem docstring.*

### Method Favela.load_points

```python
Favela.load_points()
```

*Sem docstring.*

### Method Favela.periodo

```python
Favela.periodo()
```

*Sem docstring.*

## Class Path

PurePath subclass that can make system calls.

Path represents a filesystem path but unlike PurePath, also offers
methods to do system calls on path objects. Depending on your system,
instantiating a Path will return either a PosixPath or a WindowsPath
object. You can also instantiate a PosixPath or WindowsPath directly,
but cannot instantiate a WindowsPath on a POSIX system or vice versa.

### Method Path.__bytes__

```python
Path.__bytes__()
```

Return the bytes representation of the path.  This is only
recommended to use under Unix.

### Method Path.__enter__

```python
Path.__enter__()
```

*Sem docstring.*

### Method Path.__eq__

```python
Path.__eq__()
```

Return self==value.

### Method Path.__exit__

```python
Path.__exit__()
```

*Sem docstring.*

### Method Path.__fspath__

```python
Path.__fspath__()
```

*Sem docstring.*

### Method Path.__ge__

```python
Path.__ge__()
```

Return self>=value.

### Method Path.__gt__

```python
Path.__gt__()
```

Return self>value.

### Method Path.__hash__

```python
Path.__hash__()
```

Return hash(self).

### Method Path.__le__

```python
Path.__le__()
```

Return self<=value.

### Method Path.__lt__

```python
Path.__lt__()
```

Return self<value.

### Method Path.__new__

```python
Path.__new__()
```

Construct a PurePath from one or several strings and or existing
PurePath objects.  The strings and path objects are combined so as
to yield a canonicalized path, which is incorporated into the
new PurePath object.

### Method Path.__reduce__

```python
Path.__reduce__()
```

Helper for pickle.

### Method Path.__repr__

```python
Path.__repr__()
```

Return repr(self).

### Method Path.__rtruediv__

```python
Path.__rtruediv__()
```

*Sem docstring.*

### Method Path.__str__

```python
Path.__str__()
```

Return the string representation of the path, suitable for
passing to system calls.

### Method Path.__truediv__

```python
Path.__truediv__()
```

*Sem docstring.*

### Method Path._make_child

```python
Path._make_child()
```

*Sem docstring.*

### Method Path._make_child_relpath

```python
Path._make_child_relpath()
```

*Sem docstring.*

### Method Path._scandir

```python
Path._scandir()
```

*Sem docstring.*

### Method Path.absolute

```python
Path.absolute()
```

Return an absolute version of this path by prepending the current
working directory. No normalization or symlink resolution is performed.

Use resolve() to get the canonical path to a file.

### Method Path.as_posix

```python
Path.as_posix()
```

Return the string representation of the path with forward (/)
slashes.

### Method Path.as_uri

```python
Path.as_uri()
```

Return the path as a 'file' URI.

### Method Path.chmod

```python
Path.chmod()
```

Change the permissions of the path, like os.chmod().

### Method Path.exists

```python
Path.exists()
```

Whether this path exists.

### Method Path.expanduser

```python
Path.expanduser()
```

Return a new path with expanded ~ and ~user constructs
(as returned by os.path.expanduser)

### Method Path.glob

```python
Path.glob()
```

Iterate over this subtree and yield all existing files (of any
kind, including directories) matching the given relative pattern.

### Method Path.group

```python
Path.group()
```

Return the group name of the file gid.

### Method Path.hardlink_to

```python
Path.hardlink_to()
```

Make this path a hard link pointing to the same file as *target*.

Note the order of arguments (self, target) is the reverse of os.link's.

### Method Path.is_absolute

```python
Path.is_absolute()
```

True if the path is absolute (has both a root and, if applicable,
a drive).

### Method Path.is_block_device

```python
Path.is_block_device()
```

Whether this path is a block device.

### Method Path.is_char_device

```python
Path.is_char_device()
```

Whether this path is a character device.

### Method Path.is_dir

```python
Path.is_dir()
```

Whether this path is a directory.

### Method Path.is_fifo

```python
Path.is_fifo()
```

Whether this path is a FIFO.

### Method Path.is_file

```python
Path.is_file()
```

Whether this path is a regular file (also True for symlinks pointing
to regular files).

### Method Path.is_mount

```python
Path.is_mount()
```

Check if this path is a POSIX mount point

### Method Path.is_relative_to

```python
Path.is_relative_to()
```

Return True if the path is relative to another path or False.
        

### Method Path.is_reserved

```python
Path.is_reserved()
```

Return True if the path contains one of the special names reserved
by the system, if any.

### Method Path.is_socket

```python
Path.is_socket()
```

Whether this path is a socket.

### Method Path.is_symlink

```python
Path.is_symlink()
```

Whether this path is a symbolic link.

### Method Path.iterdir

```python
Path.iterdir()
```

Iterate over the files in this directory.  Does not yield any
result for the special paths '.' and '..'.

### Method Path.joinpath

```python
Path.joinpath()
```

Combine this path with one or several arguments, and return a
new path representing either a subpath (if all arguments are relative
paths) or a totally different path (if one of the arguments is
anchored).

### Method Path.lchmod

```python
Path.lchmod()
```

Like chmod(), except if the path points to a symlink, the symlink's
permissions are changed, rather than its target's.

### Method Path.link_to

```python
Path.link_to()
```

Make the target path a hard link pointing to this path.

Note this function does not make this path a hard link to *target*,
despite the implication of the function and argument names. The order
of arguments (target, link) is the reverse of Path.symlink_to, but
matches that of os.link.

Deprecated since Python 3.10 and scheduled for removal in Python 3.12.
Use `hardlink_to()` instead.

### Method Path.lstat

```python
Path.lstat()
```

Like stat(), except if the path points to a symlink, the symlink's
status information is returned, rather than its target's.

### Method Path.match

```python
Path.match()
```

Return True if this path matches the given pattern.

### Method Path.mkdir

```python
Path.mkdir()
```

Create a new directory at this given path.

### Method Path.open

```python
Path.open()
```

Open the file pointed by this path and return a file object, as
the built-in open() function does.

### Method Path.owner

```python
Path.owner()
```

Return the login name of the file owner.

### Method Path.read_bytes

```python
Path.read_bytes()
```

Open the file in bytes mode, read it, and close the file.

### Method Path.read_text

```python
Path.read_text()
```

Open the file in text mode, read it, and close the file.

### Method Path.readlink

```python
Path.readlink()
```

Return the path to which the symbolic link points.

### Method Path.relative_to

```python
Path.relative_to()
```

Return the relative path to another path identified by the passed
arguments.  If the operation is not possible (because this is not
a subpath of the other path), raise ValueError.

### Method Path.rename

```python
Path.rename()
```

Rename this path to the target path.

The target path may be absolute or relative. Relative paths are
interpreted relative to the current working directory, *not* the
directory of the Path object.

Returns the new Path instance pointing to the target path.

### Method Path.replace

```python
Path.replace()
```

Rename this path to the target path, overwriting if that path exists.

The target path may be absolute or relative. Relative paths are
interpreted relative to the current working directory, *not* the
directory of the Path object.

Returns the new Path instance pointing to the target path.

### Method Path.resolve

```python
Path.resolve()
```

Make the path absolute, resolving all symlinks on the way and also
normalizing it.

### Method Path.rglob

```python
Path.rglob()
```

Recursively yield all existing files (of any kind, including
directories) matching the given relative pattern, anywhere in
this subtree.

### Method Path.rmdir

```python
Path.rmdir()
```

Remove this directory.  The directory must be empty.

### Method Path.samefile

```python
Path.samefile()
```

Return whether other_path is the same or not as this file
(as returned by os.path.samefile()).

### Method Path.stat

```python
Path.stat()
```

Return the result of the stat() system call on this path, like
os.stat() does.

### Method Path.symlink_to

```python
Path.symlink_to()
```

Make this path a symlink pointing to the target path.
Note the order of arguments (link, target) is the reverse of os.symlink.

### Method Path.touch

```python
Path.touch()
```

Create this file with the given access mode, if it doesn't exist.

### Method Path.unlink

```python
Path.unlink()
```

Remove this file or link.
If the path is a directory, use rmdir() instead.

### Method Path.with_name

```python
Path.with_name()
```

Return a new path with the file name changed.

### Method Path.with_stem

```python
Path.with_stem()
```

Return a new path with the stem changed.

### Method Path.with_suffix

```python
Path.with_suffix()
```

Return a new path with the file suffix changed.  If the path
has no suffix, add given suffix.  If the given suffix is an empty
string, remove the suffix from the path.

### Method Path.write_bytes

```python
Path.write_bytes()
```

Open the file in bytes mode, write to it, and close the file.

### Method Path.write_text

```python
Path.write_text()
```

Open the file in text mode, write to it, and close the file.
