#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/numerics/data_out.h>

#include <mpi.h>

// Simple program for setting up a Triangulation on one processor and
// broadcasting it to the whole network.

using namespace dealii;

namespace
{
  void
  scatter_triangulation(const AlignedVector<char> &tria_buffer,
                        types::subdomain_id        n_splits,
                        parallel::fullydistributed::Triangulation<3> &tria)
  {
    Assert(n_splits > 0,
           ExcMessage("There must be at least one split per node."));
    const MPI_Comm communicator = tria.get_mpi_communicator();

    // There is no guarantee that ranks are consecutive on a node, so to fairly
    // split things we have to get a per-node numbering
    MPI_Comm node_communicator;
    int      ierr = MPI_Comm_split_type(
      communicator, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_communicator);
    AssertThrowMPI(ierr);
    const types::subdomain_id rank =
      Utilities::MPI::this_mpi_process(node_communicator);
    const types::subdomain_id n_ranks =
      Utilities::MPI::n_mpi_processes(node_communicator);
    const types::subdomain_id group_size =
      std::max(std::min(n_ranks / n_splits, n_ranks), 1u);
    Assert(group_size > 0, ExcInternalError());

    const types::subdomain_id global_rank =
      Utilities::MPI::this_mpi_process(communicator);
    const std::vector<types::subdomain_id> global_ranks =
      Utilities::MPI::all_gather(node_communicator, global_rank);

    const int size_tag   = 10;
    const int buffer_tag = 11;

    TriangulationDescription::Description<3> description;
    if (rank % group_size == 0)
      {
        Triangulation<3> serial_tria;

        boost::iostreams::array_source tria_in(tria_buffer.begin(),
                                               tria_buffer.end());
        boost::iostreams::stream_buffer<decltype(tria_in)> tria_stream_buffer(
          tria_in);
        boost::archive::binary_iarchive tria_iarchive(tria_stream_buffer);
        serial_tria.load(tria_iarchive, 0u);

        std::deque<std::vector<char>> send_buffers;
        std::deque<unsigned long>     send_sizes;
        std::vector<MPI_Request>      send_requests;
        send_requests.reserve(2 * group_size);
        for (types::subdomain_id group_offset = 1; group_offset < group_size;
             ++group_offset)
          {
            if (rank + group_offset >= n_ranks)
              break;
            AssertIndexRange(rank + group_offset, global_ranks.size());
            const auto other_description = TriangulationDescription::Utilities::
              create_description_from_triangulation(
                serial_tria,
                communicator,
                {},
                global_ranks[rank + group_offset]);
            send_buffers.emplace_back();
            std::vector<char> &buffer = send_buffers.back();
            Utilities::pack(other_description, buffer, false);
            send_sizes.push_back(buffer.size());
            const unsigned long &size = send_sizes.back();

            send_requests.emplace_back();
            ierr = MPI_Isend(&size,
                             1,
                             MPI_UNSIGNED_LONG,
                             rank + group_offset,
                             size_tag,
                             node_communicator,
                             &send_requests.back());
            AssertThrowMPI(ierr);
            send_requests.emplace_back();
            ierr = MPI_Isend(buffer.data(),
                             static_cast<int>(buffer.size()),
                             MPI_BYTE,
                             rank + group_offset,
                             buffer_tag,
                             node_communicator,
                             &send_requests.back());
            AssertThrowMPI(ierr);
          }

        AssertIndexRange(rank, global_ranks.size());
        description = TriangulationDescription::Utilities::
          create_description_from_triangulation(serial_tria,
                                                communicator,
                                                {},
                                                global_ranks[rank]);

        ierr = MPI_Waitall(send_requests.size(),
                           send_requests.data(),
                           MPI_STATUSES_IGNORE);
        AssertThrowMPI(ierr);
      }
    else
      {
        const int     group_root_rank = (rank / group_size) * group_size;
        unsigned long size = std::numeric_limits<unsigned long>::max();

        ierr = MPI_Recv(&size,
                        1,
                        MPI_UNSIGNED_LONG,
                        group_root_rank,
                        size_tag,
                        node_communicator,
                        MPI_STATUS_IGNORE);
        AssertThrowMPI(ierr);
        std::vector<char> buffer(size);
        ierr = MPI_Recv(buffer.data(),
                        static_cast<int>(buffer.size()),
                        MPI_BYTE,
                        group_root_rank,
                        buffer_tag,
                        node_communicator,
                        MPI_STATUS_IGNORE);
        AssertThrowMPI(ierr);
        description = Utilities::unpack<decltype(description)>(buffer, false);
        description.comm = communicator;
      }

    tria.set_manifold(0, CylindricalManifold<3>());
    tria.create_triangulation(description);
    Utilities::MPI::free_communicator(node_communicator);
  }

  void
  parse_triangulation(const unsigned int n_partitions, Triangulation<3> &tria)
  {
    GridGenerator::subdivided_cylinder(tria, 10, 2.0);
    // TODO simplices
    tria.refine_global(6);
    std::cout << "Number of cells " << tria.n_active_cells() << std::endl;
    GridTools::partition_triangulation_zorder(n_partitions, tria);
  }

  /**
   * AlignedVector::insert() only allocates enough room to add the new
   * entries, instead of doubling the capacity. Work around it by using
   * insert_back() instead.
   */
  template <typename T>
  struct AlignedVectorInserter
  {
    AlignedVectorInserter(AlignedVector<T> &vec)
      : m_vec(&vec)
    {}

    using value_type     = typename AlignedVector<T>::value_type;
    using const_iterator = typename AlignedVector<T>::const_iterator;
    using iterator       = typename AlignedVector<T>::iterator;

    void
    insert(iterator begin, const_iterator input_begin, const_iterator input_end)
    {
      Assert(begin == m_vec->end(), ExcCardinalInternalError());
      m_vec->insert_back(input_begin, input_end);
    }

    iterator
    end()
    {
      return m_vec->end();
    }

    const_iterator
    end() const
    {
      return m_vec->end();
    }

    AlignedVector<T> *m_vec;
  };
} // namespace

void
parse_triangulation_input(parallel::fullydistributed::Triangulation<3> &tria)
{
  const MPI_Comm communicator = tria.get_mpi_communicator();
  const auto     n_partitions = Utilities::MPI::n_mpi_processes(communicator);
  AlignedVector<char> tria_buffer;
  if (Utilities::MPI::this_mpi_process(communicator) == 0u)
    {
      Triangulation<3> serial_tria;
      parse_triangulation(n_partitions, serial_tria);
      AlignedVectorInserter<char> tria_buffer_inserter(tria_buffer);
      auto tria_stream = boost::iostreams::back_inserter(tria_buffer_inserter);
      boost::iostreams::stream_buffer<decltype(tria_stream)> tria_stream_buffer(
        tria_stream);
      boost::archive::binary_oarchive tria_oarchive(tria_stream_buffer);
      serial_tria.save(tria_oarchive, 0u);
    }

  int ierr = MPI_Barrier(communicator);
  AssertThrowMPI(ierr);
  tria_buffer.replicate_across_communicator(communicator, 0u);

  scatter_triangulation(tria_buffer, 2, tria);
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init_finalize(argc, argv);

  parallel::fullydistributed::Triangulation<3> tria(MPI_COMM_WORLD);

  parse_triangulation_input(tria);

  DataOut<3> data_out;
  data_out.attach_triangulation(tria);

  Vector<float> subdomain(tria.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = tria.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record(
    "./", "solution", 0, MPI_COMM_WORLD, 2, 1);
}
